"""Per-fold / mean+-std / pooled metrics, ROC and training-curve plots, JSON dump."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as skm


def _scores(y, o, p):
    fpr, tpr, _ = skm.roc_curve(y, p, pos_label=1)
    return dict(acc=skm.accuracy_score(y, o), auc=skm.auc(fpr, tpr),
                sen=skm.recall_score(y, o, pos_label=1, zero_division=0),
                spe=skm.recall_score(y, o, pos_label=0, zero_division=0),
                f1=skm.f1_score(y, o, pos_label=1, zero_division=0))


def report(results, cfg, out_dir, title):
    os.makedirs(out_dir, exist_ok=True)
    K = len(results)
    per = {s: [] for s in ("val", "test")}
    for r in results:
        per["val"].append(_scores(r["val_t"], r["val_o"], r["val_p"]))
        per["test"].append(_scores(r["test_t"], r["test_o"], r["test_p"]))

    print("\n" + "=" * 130 + f"\nPER-FOLD RESULTS  ({title})\n" + "=" * 130)
    for i, r in enumerate(results):
        v, t = per["val"][i], per["test"][i]
        print(f"Fold[{i + 1}/{K}] | Train CE {r['train_loss']:.3f} ACC {r['train_acc'] * 100:.2f}% | "
              f"Val ACC {v['acc'] * 100:.2f}% AUC {v['auc']:.4f} SEN {v['sen']:.4f} SPE {v['spe']:.4f} F1 {v['f1']:.4f} | "
              f"Test ACC {t['acc'] * 100:.2f}% AUC {t['auc']:.4f} SEN {t['sen']:.4f} SPE {t['spe']:.4f} F1 {t['f1']:.4f}")

    summary = {"title": title, "config": {k: v for k, v in vars(cfg).items()}, "folds": [], "mean_std": {}, "pooled": {}}
    for i, r in enumerate(results):
        summary["folds"].append(dict(fold=i + 1, best_epoch=r["epoch"], train_loss=r["train_loss"],
                                     train_acc=r["train_acc"], val=per["val"][i], test=per["test"][i]))

    print("\n" + "=" * 60 + f"\nMEAN +/- STD ACROSS {K} FOLDS\n" + "=" * 60)
    print(f"Train CE:       {np.mean([r['train_loss'] for r in results]):.4f} +/- {np.std([r['train_loss'] for r in results]):.4f}")
    print(f"Train Accuracy: {np.mean([r['train_acc'] for r in results]):.4f} +/- {np.std([r['train_acc'] for r in results]):.4f}")
    for split in ("val", "test"):
        for m, name in (("acc", "Accuracy"), ("auc", "AUC"), ("sen", "Sensitivity"), ("spe", "Specificity"), ("f1", "F1")):
            vals = [d[m] for d in per[split]]
            summary["mean_std"][f"{split}_{m}"] = (float(np.mean(vals)), float(np.std(vals)))
            print(f"{split.capitalize():5s} {name:12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
        print("-" * 60)

    cat = lambda key: np.concatenate([r[key] for r in results])
    for split in ("val", "test"):
        s = _scores(cat(f"{split}_t"), cat(f"{split}_o"), cat(f"{split}_p"))
        summary["pooled"][split] = s
        print(f"POOLED {split:5s} -> ACC {s['acc']:.4f} | AUC {s['auc']:.4f} | SEN {s['sen']:.4f} | SPE {s['spe']:.4f} | F1 {s['f1']:.4f}")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ---- plots (saved as PNG)
    plt.figure(figsize=(7, 6))
    plt.grid()
    for i, r in enumerate(results):
        fpr, tpr, _ = skm.roc_curve(r["test_t"], r["test_p"], pos_label=1)
        plt.plot(fpr, tpr, lw=1.5, label=f"Fold {i + 1} (AUC={per['test'][i]['auc']:.3f})")
    fpr, tpr, _ = skm.roc_curve(cat("test_t"), cat("test_p"), pos_label=1)
    plt.plot(fpr, tpr, "k", lw=2.5, label=f"Pooled (AUC={summary['pooled']['test']['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"Test ROC - {title}")
    plt.legend(loc="lower right"); plt.savefig(os.path.join(out_dir, "roc_test.png"), dpi=130, bbox_inches="tight"); plt.close()

    for i, r in enumerate(results):
        h = r["history"]
        x = range(1, len(h["val_acc"]) + 1)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(x, h["train_acc"], "b", label="train"); ax[0].plot(x, h["val_acc"], "r", label="val"); ax[0].plot(x, h["test_acc"], "g", label="test")
        ax[0].set_title(f"Accuracy - fold {i + 1}"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(x, h["train_loss"], "b", label="train CE"); ax[1].plot(x, h["val_loss"], "r", label="val"); ax[1].plot(x, h["test_loss"], "g", label="test")
        ax[1].set_title(f"Cross-entropy - fold {i + 1}"); ax[1].set_xlabel("epoch"); ax[1].legend()
        fig.savefig(os.path.join(out_dir, f"curves_fold{i + 1}.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved results.json, roc_test.png, curves_fold*.png in {out_dir}")
    return summary
