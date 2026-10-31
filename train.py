"""ONE training loop for sMRI, fMRI and fusion -> identical optimiser, batch size, epochs, mixup,
clipping, seeds and model-selection rule for all three."""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics as skm

from models import FMRIModel, FusionNet, GraphBank, SMRIClassifier


# ------------------------------------------------------------------ mixup helpers
def mixed_ce(logits, ya, yb, lam):
    return lam * F.cross_entropy(logits, ya) + (1 - lam) * F.cross_entropy(logits, yb)


def mixup_cluster_loss(m, ya, yb, lam, intra_weight=2):
    """DHGFormer group loss (pulls same-class adjacency matrices together, pushes classes apart)."""
    y1 = lam * ya.float() + (1 - lam) * yb.float()
    y0 = 1 - y1
    bz, r, _ = m.shape
    m = m.reshape(bz, -1)
    s1, s0, loss = torch.sum(y1), torch.sum(y0), 0.0
    if s0 > 0:
        c0 = torch.matmul(y0, m) / s0
        loss = loss + torch.matmul(y0, torch.norm(m - c0, p=1, dim=1)) / (s0 * r * r)
    if s1 > 0:
        c1 = torch.matmul(y1, m) / s1
        loss = loss + torch.matmul(y1, torch.norm(m - c1, p=1, dim=1)) / (s1 * r * r)
    if s0 > 0 and s1 > 0:
        loss = loss + intra_weight * (1 - torch.norm(c0 - c1, p=1) / (r * r))
    return loss


def mix_batch(batch, perm, lam):
    """Same permutation and lambda for every modality of the batch."""
    out = dict(batch)
    if batch.get("xs") is not None:
        out["xs"] = [lam * x + (1 - lam) * x[perm] for x in batch["xs"]]
    if batch.get("ts") is not None:
        out["ts"] = lam * batch["ts"] + (1 - lam) * batch["ts"][perm]
        out["corr"] = lam * batch["corr"] + (1 - lam) * batch["corr"][perm]
    return out


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ one fold
def run_fold(cfg, cohort, fold, split, smri_data, cluster_order, dev, save_dir):
    tr, va, te = split["train"], split["val"], split["test"]
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te)), "folds overlap!"
    use_s = cfg.modality in ("smri", "both")
    use_f = cfg.modality in ("fmri", "both")
    set_seed(cfg.seed + fold)
    y_all = torch.from_numpy(cohort.labels)

    X = bank = ts_all = corr_all = None
    in_channels, base_ew, fmri_len = None, None, None
    if use_s:
        X = [torch.from_numpy(x).to(dev) for x in smri_data["X"]]
        in_channels = [x.shape[2] for x in X]
        base_ew = smri_data["edge_weight"]
        bank = GraphBank(smri_data["edge_index"], smri_data["edge_weight"], smri_data["n_nodes"], dev)
    if use_f:
        mu = cohort.fmri_ts[tr].mean(dtype=np.float64)                # scaling statistics: TRAIN only
        sd = cohort.fmri_ts[tr].std(dtype=np.float64)
        ts_all = torch.from_numpy(((cohort.fmri_ts - mu) / (sd + 1e-8)).astype(np.float32))
        corr_all = torch.from_numpy(cohort.fmri_corr)
        fmri_len = ts_all.shape[2]

    if cfg.modality == "smri":
        model = SMRIClassifier(in_channels, cfg, base_ew)
    elif cfg.modality == "fmri":
        model = FMRIModel(cfg, fmri_len, cluster_order)
    else:
        model = FusionNet(in_channels, cfg, base_ew, cluster_order, fmri_len)
    model = model.to(dev)

    # ONE lr / weight decay for every parameter (learnable edge weights, if used, get a scaled lr)
    edge_p = [p for n, p in model.named_parameters() if "edge_w" in n]
    edge_ids = {id(p) for p in edge_p}
    rest = [p for p in model.parameters() if id(p) not in edge_ids]
    groups = [dict(params=rest, lr=cfg.lr, weight_decay=cfg.weight_decay)]
    if edge_p:
        groups.append(dict(params=edge_p, lr=cfg.lr * cfg.edge_lr_scale, weight_decay=0.0))
    opt = torch.optim.Adam(groups)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Fold {fold}: train {len(tr)} | val {len(va)} | test {len(te)} | params {n_par:,}")

    def get_batch(idx):
        b = dict(m=len(idx), xs=None, bank=None, ts=None, corr=None)
        if use_s:
            ii = torch.as_tensor(idx, device=dev)
            b["xs"], b["bank"] = [x[ii] for x in X], bank
        if use_f:
            it = torch.as_tensor(idx)
            b["ts"], b["corr"] = ts_all[it].to(dev), corr_all[it].to(dev)
        return b

    @torch.no_grad()
    def evaluate(idx):
        model.eval()
        probs, loss_sum = [], 0.0
        for s in range(0, len(idx), 64):
            b = idx[s:s + 64]
            logits = model(get_batch(b))["logits"]
            loss_sum += F.cross_entropy(logits, y_all[b].to(dev), reduction="sum").item()
            probs.append(F.softmax(logits, dim=1).cpu())
        p = torch.cat(probs).numpy()
        return y_all[idx].numpy(), p.argmax(1), p, loss_sum / len(idx)

    hist = {k: [] for k in ("train_loss", "train_acc", "val_acc", "val_loss", "test_acc", "test_loss")}
    best_val, best, best_state = -1.0, {}, None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = np.random.permutation(tr)
        ce_sum, correct, n_seen = 0.0, 0.0, 0
        for s in range(0, len(order), cfg.batch_size):
            b = order[s:s + cfg.batch_size]
            if len(b) < 2:                                     # BatchNorm / group loss need >= 2 samples
                continue
            batch = get_batch(b)
            y = y_all[b].to(dev)
            if cfg.mixup:
                lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)) if cfg.mixup_alpha > 0 else 1.0
                perm = torch.randperm(len(b), device=dev)
                batch, yb = mix_batch(batch, perm, lam), y[perm]
            else:
                lam, yb = 1.0, y
            out = model(batch)
            ce_main = mixed_ce(out["logits"], y, yb, lam)
            loss = cfg.ce_weight * ce_main
            for a in out["aux"]:
                loss = loss + cfg.aux_weight * mixed_ce(a, y, yb, lam)
            if out["adj"] is not None:                         # DHGFormer regularisers
                if cfg.group_loss:
                    loss = loss + mixup_cluster_loss(out["adj"], y, yb, lam)
                if cfg.sparsity_loss:
                    loss = loss + cfg.sparsity_weight * torch.norm(out["adj"], p=1)
            opt.zero_grad()
            loss.backward()
            if all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                opt.step()
                if hasattr(model, "post_step"):
                    model.post_step()
            else:
                print("  [warning] NaN/Inf gradient -> step skipped")
            ce_sum += ce_main.item() * len(b)
            correct += (out["logits"].argmax(1) == y).float().sum().item()
            n_seen += len(b)
        train_loss, train_acc = ce_sum / n_seen, correct / n_seen

        va_t, va_o, va_p, va_l = evaluate(va)
        te_t, te_o, te_p, te_l = evaluate(te)
        va_acc, te_acc = skm.accuracy_score(va_t, va_o), skm.accuracy_score(te_t, te_o)
        for k, v in zip(hist, (train_loss, train_acc, va_acc, va_l, te_acc, te_l)):
            hist[k].append(v)

        if va_acc >= best_val:                                 # model selection on VALIDATION only
            best_val = va_acc
            best = dict(epoch=epoch, train_acc=train_acc, train_loss=train_loss,
                        val_t=va_t, val_o=va_o, val_p=va_p[:, 1],
                        test_t=te_t, test_o=te_o, test_p=te_p[:, 1], test_acc=te_acc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(f"  Epoch {epoch}: train CE {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val acc {va_acc:.4f} | test acc {te_acc:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state, os.path.join(save_dir, f"best_fold{fold}.pth"))
    print(f"Fold {fold}: best val acc {best_val:.4f} at epoch {best['epoch']} | test acc {best['test_acc']:.4f}\n")
    best["history"] = hist
    best["best_val"] = best_val
    return best
