"""Leak-free sMRI preprocessing. EVERYTHING here is fitted on the TRAIN subjects of one fold only
(impute -> ComBat -> scale -> Ridge/RFE -> per-view scaling -> covariance graph), then applied to all."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import RFE
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- views / ROI layout
ASEG_SUFFIXES = ["NVoxels", "Volume_mm3", "normMax", "normMean", "normMin", "normRange", "normStdDev"]
APARC_SUFFIXES = ["NumVert", "SurfArea", "GrayVol", "ThickAvg", "ThickStd", "MeanCurv", "GausCurv", "FoldInd", "CurvInd"]
VIEW_CONFIGS = {
    "aseg": {"prefixes": ["aseg"], "suffixes": ASEG_SUFFIXES},
    "aparc": {"prefixes": ["lh", "rh"], "suffixes": APARC_SUFFIXES},
    "wmparc": {"prefixes": ["wmparc"], "suffixes": ASEG_SUFFIXES},
}
VIEW_NAMES = list(VIEW_CONFIGS)


def _parse_roi_columns(prefix, columns, suffixes):
    sorted_suf = sorted(suffixes, key=len, reverse=True)
    roi_map = {}
    for col in columns:
        if not col.startswith(prefix + "_"):
            continue
        rem = col[len(prefix) + 1:]
        for suf in sorted_suf:
            if rem.endswith("_" + suf):
                roi_map.setdefault(rem[: -(len(suf) + 1)], {})[suf] = col
                break
    return roi_map


def build_layout(feature_cols):
    """view -> int array (n_nodes, n_subfeat) of column indices into the flat feature matrix (-1 = missing).
    Each ROI is one graph node; its sub-features (volume, thickness, ...) are the node features."""
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}
    layout = {}
    for view, vc in VIEW_CONFIGS.items():
        rows = []
        for prefix in vc["prefixes"]:
            for _, s2c in _parse_roi_columns(prefix, feature_cols, vc["suffixes"]).items():
                rows.append([col_to_idx[s2c[s]] if s in s2c else -1 for s in vc["suffixes"]])
        layout[view] = np.array(rows, dtype=np.int64)
    return layout


# --------------------------------------------------------------------------- ComBat (fit on train)
def build_combat_covariates(pheno, cat_cols, cont_cols):
    cols = []
    for c in cat_cols:
        d = pd.get_dummies(pheno[c].astype(str).reset_index(drop=True), drop_first=True).values.astype(np.float64)
        if d.shape[1]:
            cols.append(d)
    if cont_cols:
        cont = pheno[cont_cols].reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
        cols.append(cont.fillna(cont.mean()).values.astype(np.float64))
    return np.concatenate(cols, axis=1) if cols else np.zeros((len(pheno), 0))


def combat_fit(data, batch, mod, eps=1e-8):
    data = np.asarray(data, dtype=np.float64)
    n, _ = data.shape
    batch = np.asarray(batch).astype(str)
    sites_u = np.unique(batch)
    nb = len(sites_u)
    for s in sites_u:
        if (batch == s).sum() < 2:
            raise ValueError(f"ComBat needs >=2 TRAIN subjects per site; site {s!r} has fewer.")
    B = np.stack([(batch == s).astype(np.float64) for s in sites_u], axis=1)
    design = np.concatenate([B, mod], axis=1)
    ok = data.std(axis=0) > eps
    X = data[:, ok]

    beta = np.linalg.lstsq(design, X, rcond=None)[0]
    grand_mean = (B.sum(axis=0) / n) @ beta[:nb]
    resid = X - design @ beta
    var_pooled = np.maximum((resid ** 2).mean(axis=0), eps)
    stand_mean = grand_mean[None, :] + (mod @ beta[nb:] if mod.shape[1] else 0.0)
    s_data = (X - stand_mean) / np.sqrt(var_pooled)[None, :]

    gamma_hat = np.stack([s_data[batch == s].mean(axis=0) for s in sites_u])
    delta_hat = np.maximum(np.stack([s_data[batch == s].var(axis=0, ddof=1) for s in sites_u]), eps)
    gamma_bar = gamma_hat.mean(axis=1)
    t2 = gamma_hat.var(axis=1, ddof=1)
    m = delta_hat.mean(axis=1)
    s2 = np.maximum(delta_hat.var(axis=1, ddof=1), eps)
    a_prior = (2 * s2 + m ** 2) / s2
    b_prior = (m * s2 + m ** 3) / s2

    gamma_star, delta_star = np.zeros_like(gamma_hat), np.zeros_like(delta_hat)
    for i, s in enumerate(sites_u):
        xb = s_data[batch == s]
        n_i = xb.shape[0]
        g_old, d_old = gamma_hat[i].copy(), delta_hat[i].copy()
        for _ in range(500):
            g_new = (t2[i] * n_i * gamma_hat[i] + d_old * gamma_bar[i]) / (t2[i] * n_i + d_old)
            sum2 = ((xb - g_new[None, :]) ** 2).sum(axis=0)
            d_new = (0.5 * sum2 + b_prior[i]) / (n_i / 2.0 + a_prior[i] - 1.0)
            change = max(np.max(np.abs(g_new - g_old) / (np.abs(g_old) + eps)),
                         np.max(np.abs(d_new - d_old) / (np.abs(d_old) + eps)))
            g_old, d_old = g_new, d_new
            if change < 1e-4:
                break
        gamma_star[i], delta_star[i] = g_old, d_old
    return dict(sites_u=list(sites_u), nb=nb, ok=ok, beta=beta, grand_mean=grand_mean,
                var_pooled=var_pooled, gamma_star=gamma_star, delta_star=delta_star)


def combat_apply(data, batch, mod, P):
    data = np.asarray(data, dtype=np.float64)
    batch = np.asarray(batch).astype(str)
    ok, nb = P["ok"], P["nb"]
    X = data[:, ok]
    stand_mean = P["grand_mean"][None, :] + (mod @ P["beta"][nb:] if mod.shape[1] else 0.0)
    s_data = (X - stand_mean) / np.sqrt(P["var_pooled"])[None, :]
    out = s_data.copy()
    for i, s in enumerate(P["sites_u"]):
        idx = batch == s
        if idx.any():
            out[idx] = (s_data[idx] - P["gamma_star"][i][None, :]) / np.sqrt(P["delta_star"][i])[None, :]
    out = out * np.sqrt(P["var_pooled"])[None, :] + stand_mean
    result = data.copy()
    result[:, ok] = out
    return result


# --------------------------------------------------------------------------- flat-matrix preprocessing
def _make_imputer(cfg):
    kw = dict(keep_empty_features=True)
    try:
        if cfg.impute == "mean":
            return SimpleImputer(strategy="mean", **kw)
        if cfg.impute == "median":
            return SimpleImputer(strategy="median", **kw)
        if cfg.impute == "knn":
            return KNNImputer(n_neighbors=cfg.knn_neighbors, weights="distance", **kw)
    except TypeError:       # old scikit-learn without keep_empty_features
        if cfg.impute in ("mean", "median"):
            return SimpleImputer(strategy=cfg.impute)
        return KNNImputer(n_neighbors=cfg.knn_neighbors, weights="distance")
    if cfg.impute == "iterative":
        return IterativeImputer(max_iter=10, random_state=0)
    raise ValueError(f"unknown impute strategy {cfg.impute!r}")


def preprocess_flat(cohort, cfg, train_idx, combat_mod):
    X = cohort.smri_raw.copy()
    X = _make_imputer(cfg).fit(X[train_idx]).transform(X)
    if cfg.combat:
        P = combat_fit(X[train_idx], cohort.sites[train_idx], combat_mod[train_idx])
        X = combat_apply(X, cohort.sites, combat_mod, P)
    X = StandardScaler().fit(X[train_idx]).transform(X)

    selected = None
    if 0 < cfg.n_select_features < X.shape[1]:
        rfe = RFE(RidgeClassifier(), n_features_to_select=cfg.n_select_features, step=100)
        rfe.fit(X[train_idx], cohort.labels[train_idx])             # TRAIN labels only
        selected = np.where(rfe.support_)[0]
    return X, selected


def build_view_features(X, selected, layout, train_idx):
    """(N, n_nodes, n_subfeat) per view. Columns not selected by RFE (or absent) are zeroed."""
    N = X.shape[0]
    feats = {}
    for view, col_idx in layout.items():
        n_nodes, n_sub = col_idx.shape
        valid = col_idx >= 0
        if selected is not None:
            valid &= np.isin(col_idx, selected)
        mat = np.full((N, n_nodes, n_sub), np.nan)
        rr, cc = np.where(valid)
        mat[:, rr, cc] = X[:, col_idx[rr, cc]]
        flat = mat.reshape(N, -1)
        flat[:, np.all(np.isnan(flat), axis=0)] = 0.0
        flat = StandardScaler().fit(flat[train_idx]).transform(flat)   # per-view scaling, train only
        feats[view] = flat.reshape(N, n_nodes, n_sub).astype(np.float32)
    return feats


# --------------------------------------------------------------------------- covariance graph
def compute_similarity(node_feats, metric):
    if metric == "pearson":
        sim = np.corrcoef(node_feats.T)
    elif metric == "spearman":
        sim, _ = spearmanr(node_feats)
        if node_feats.shape[1] == 2:
            sim = np.array([[1.0, sim], [sim, 1.0]])
    elif metric == "cosine":
        sim = cosine_similarity(node_feats.T)
    elif metric == "euclidean":
        dist = squareform(pdist(node_feats.T, metric="euclidean"))
        sigma = np.median(dist[dist > 0])
        sim = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    elif metric == "partial_correlation":
        precision = LedoitWolf().fit(node_feats).precision_
        d = np.sqrt(np.diag(precision))
        d[d == 0] = 1e-8
        sim = -precision / np.outer(d, d)
    else:
        raise ValueError(f"unknown graph metric {metric!r}")
    return np.nan_to_num(sim, nan=0.0)


def build_covariance_graph(feats, train_idx, k, metric):
    """kNN graph over the ROI nodes of one view, using TRAIN subjects x all sub-features only."""
    f = feats[train_idx]                                            # (n_train, n_nodes, n_sub)
    node_feats = f.transpose(0, 2, 1).reshape(-1, f.shape[1])       # (n_train * n_sub, n_nodes)
    sim = compute_similarity(node_feats, metric)
    np.fill_diagonal(sim, -np.inf)
    n = sim.shape[0]
    k = min(k, n - 1)
    nbr = np.argsort(sim, axis=1)[:, -k:]
    src = np.repeat(np.arange(n), k)
    dst = nbr.ravel()
    w = sim[src, dst].astype(np.float32)
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    return edge_index, np.concatenate([w, w])


def prepare_fold(cohort, cfg, layout, combat_mod, train_idx, verbose=True):
    """Everything the sMRI branch needs for one fold."""
    if verbose:
        print(f"  sMRI preprocessing on {len(train_idx)} train subjects (impute/ComBat/scale/RFE/graph)")
    X, selected = preprocess_flat(cohort, cfg, train_idx, combat_mod)
    feats = build_view_features(X, selected, layout, train_idx)
    out = dict(X=[], edge_index=[], edge_weight=[], n_nodes=[])
    for view in VIEW_NAMES:
        k = cfg.k_neighbors.get(view, cfg.default_k_neighbors)
        ei, ew = build_covariance_graph(feats[view], train_idx, k, cfg.graph_metric)
        out["X"].append(feats[view])
        out["edge_index"].append(ei)
        out["edge_weight"].append(ew)
        out["n_nodes"].append(feats[view].shape[1])
    return out
