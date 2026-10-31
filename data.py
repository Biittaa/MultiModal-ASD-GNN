"""Subject table, fMRI alignment and the (site-stratified) cross-validation folds."""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass
class Cohort:
    ids: list                    # FILE_IDs
    sites: np.ndarray            # str, spaces removed
    labels: np.ndarray           # int64, 1 = autism, 0 = control
    pheno: pd.DataFrame          # indexed by FILE_ID, same order as ids
    smri_raw: np.ndarray         # (N, n_features) float64, may contain NaN
    feature_cols: list
    fmri_ts: np.ndarray = None   # (N, 200, T) float32
    fmri_corr: np.ndarray = None  # (N, 200, 200) float32
    usable: np.ndarray = None    # (N,) bool - subject has the data the chosen modality needs


def load_cohort(cfg):
    pheno = pd.read_csv(cfg.pheno_csv)
    pheno["FILE_ID"] = pheno["FILE_ID"].astype(str)
    print(f"Phenotypic file: {len(pheno)} rows")
    pheno = pheno[pheno["FILE_ID"] != "no_filename"]
    if cfg.apply_sub_in_smp_filter and "SUB_IN_SMP" in pheno.columns:
        pheno = pheno[pheno["SUB_IN_SMP"] == 1]

    smri = pd.read_csv(cfg.smri_csv)
    smri["subject_id"] = smri["subject_id"].astype(str)
    print(f"sMRI feature file: {len(smri)} subjects")

    common = sorted(set(pheno["FILE_ID"]) & set(smri["subject_id"]))
    print(f"{len(common)} subjects in both files")
    pheno = pheno.set_index("FILE_ID").loc[common]
    smri = smri.set_index("subject_id").loc[common]

    sites = np.array(pheno["SITE_ID"].astype(str).str.replace(" ", "").tolist())
    labels = (pheno["DX_GROUP"].values == 1).astype(np.int64)      # DX_GROUP 1 = autism
    feature_cols = list(smri.columns)
    smri_raw = smri[feature_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
    print(f"sMRI matrix {smri_raw.shape}, NaNs: {int(np.isnan(smri_raw).sum())}")
    return Cohort(list(common), sites, labels, pheno, smri_raw, feature_cols,
                  usable=np.ones(len(common), dtype=bool))


def _read_order(path):
    """subject_order.txt: rows 'index_in_drive subject_id [site]' (header / === lines skipped)
    or simply one id per line. Returns [(row_in_npy, id, site|None)]."""
    rows = []
    with open(path) as f:
        for ln in f:
            t = ln.replace(",", " ").split()
            if not t:
                continue
            if t[0].isdigit() and len(t) >= 2:
                rows.append((int(t[0]), t[1], " ".join(t[2:]) if len(t) > 2 else None))
            elif len(t) == 1 and not set(t[0]) <= set("=-"):
                rows.append((len(rows), os.path.splitext(t[0])[0], None))
    return rows


def attach_fmri(cfg, cohort):
    """Loads DHGFormer's abide.npy and aligns its rows to the cohort (by FILE_ID or SUB_ID)."""
    z = np.load(cfg.fmri_npy, allow_pickle=True).item()
    ts_raw = np.asarray(z["timeseires"])
    corr_raw = np.asarray(z["corr"])
    lab_raw = np.asarray(z["label"]).ravel()
    n_roi, n_t = ts_raw.shape[1], ts_raw.shape[2]
    print(f"abide.npy: {len(ts_raw)} subjects, time-series {ts_raw.shape[1:]}")

    rows = _read_order(cfg.fmri_order_txt)
    assert len(rows) == len(ts_raw) == len(corr_raw) == len(lab_raw), \
        f"abide.npy has {len(ts_raw)} rows but subject_order.txt lists {len(rows)}"
    assert sorted(r[0] for r in rows) == list(range(len(ts_raw))), "index_in_drive must be 0..N-1"

    cands = {"FILE_ID": {fid: i for i, fid in enumerate(cohort.ids)}}
    if "SUB_ID" in cohort.pheno.columns:
        cands["SUB_ID"] = {str(int(s)): i for i, s in enumerate(cohort.pheno["SUB_ID"].values)}
    best = max(cands, key=lambda k: sum(r[1] in cands[k] for r in rows))
    id_map = cands[best]
    n_match = sum(r[1] in id_map for r in rows)
    print(f"subject_order ids matched as {best}: {n_match}/{len(rows)}")
    if n_match == 0:
        raise ValueError("No id in subject_order.txt matches FILE_ID / SUB_ID")

    npy_lab = 1 - cohort.labels            # DHGFormer: 0 = autism, 1 = control
    by_subj = {}
    for ridx, sid, _ in rows:
        if sid in id_map:
            by_subj.setdefault(id_map[sid], []).append(ridx)

    N = len(cohort.ids)
    ts = np.zeros((N, n_roi, n_t), dtype=np.float32)
    corr = np.zeros((N, n_roi, n_roi), dtype=np.float32)
    ok = np.zeros(N, dtype=bool)
    chosen = {}
    for j, ridxs in by_subj.items():
        if len(ridxs) > 1:                 # same id on several rows: keep the single row whose label agrees
            good = [r for r in ridxs if lab_raw[r] == npy_lab[j]]
            if len(good) != 1:
                print(f"  WARNING: subject {cohort.ids[j]} ambiguous (rows {ridxs}) -> excluded")
                continue
            ridxs = good
        r = ridxs[0]
        ts[j], corr[j], ok[j], chosen[j] = ts_raw[r], corr_raw[r], True, r

    agree = float(np.mean([lab_raw[r] == npy_lab[j] for j, r in chosen.items()]))
    print(f"label agreement abide.npy <-> phenotypic: {agree * 100:.1f}% over {len(chosen)} subjects")
    assert agree > 0.99, "LABELS DISAGREE -> subject_order.txt does not match the row order of abide.npy"
    site_of_row = {r[0]: r[2] for r in rows}
    chk = [(site_of_row[r] or "").replace(" ", "") == cohort.sites[j] for j, r in chosen.items() if site_of_row[r]]
    if chk:
        print(f"site agreement: {np.mean(chk) * 100:.1f}%")
    print(f"fMRI available for {int(ok.sum())}/{N} subjects")

    cohort.fmri_ts, cohort.fmri_corr = ts, corr
    cohort.usable = ok.copy()


def make_folds(cohort, k):
    """Same scheme as the original notebook: StratifiedKFold(random_state=0) inside every site.
    Fold f: validation = chunk f, test = chunk f+1 (mod k), train = the rest.
    Subjects without the required data are removed AFTER splitting, so everyone else keeps the
    fold they had before."""
    parts = [[] for _ in range(k)]
    for s in np.unique(cohort.sites):
        idx = np.where(cohort.sites == s)[0]
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        for g, (_, val) in enumerate(skf.split(idx, cohort.labels[idx])):
            parts[g].extend(idx[val].tolist())

    keep = lambda lst: np.array([i for i in lst if cohort.usable[i]], dtype=np.int64)
    folds = {}
    for f in range(k):
        te_g = (f + 1) % k
        train = [i for g in range(k) if g not in (f, te_g) for i in parts[g]]
        folds[f + 1] = dict(train=keep(train), val=keep(parts[f]), test=keep(parts[te_g]))
    return folds
