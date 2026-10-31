"""All settings live here. Every field is also a command-line flag (see main.py),
e.g.  python main.py --modality fmri --no-mixup --lr 5e-4
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    smri_csv: str = "/content/drive/MyDrive/DHGFormer/abide_smri.csv"
    pheno_csv: str = "/content/drive/MyDrive/DHGFormer/Phenotypic_V1_0b_preprocessed1.csv"
    fmri_npy: str = "/content/drive/MyDrive/DHGFormer/abide.npy"
    fmri_order_txt: str = "/content/drive/MyDrive/DHGFormer/subject_order.txt"
    cluster_map_path: str = ""          # optional node_clus_map.pickle ("" -> built-in copy)
    out_dir: str = "/content/drive/MyDrive/sMRI"

    # ------------------------------------------------------------- experiment
    modality: str = "both"              # 'smri' | 'fmri' | 'both'
    fusion: str = "cross_attention"     # 'concat' | 'cross_attention'   (only for modality='both')
    k_fold: int = 5
    seed: int = 21                      # fold f uses seed + f, for every modality
    # Keep only subjects that have BOTH sMRI and fMRI, also when training sMRI alone,
    # so that smri / fmri / both are evaluated on exactly the same subjects and folds.
    match_fmri_subjects: bool = True
    apply_sub_in_smp_filter: bool = False

    # ------------------------------------- optimisation: SHARED by all modalities
    epochs: int = 100
    batch_size: int = 16                # mini-batch size (sMRI, fMRI and fusion alike)
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    ce_weight: float = 1.0              # weight of the main cross-entropy term
    mixup: bool = True                  # mixup on/off, same switch for sMRI, fMRI and fusion
    mixup_alpha: float = 1.0
    edge_lr_scale: float = 0.3          # only used when graph_mode is 'learnable*'

    # --------------------------- DHGFormer regularisers (apply wherever DHGFormer is used)
    group_loss: bool = True
    sparsity_loss: bool = True
    sparsity_weight: float = 1e-4

    # ------------------------------------------------------------------- sMRI
    combat: bool = True
    combat_categorical: list = field(default_factory=lambda: ["SEX"])   # not a CLI flag
    combat_continuous: list = field(default_factory=lambda: ["AGE_AT_SCAN"])
    impute: str = "knn"                 # 'mean' | 'median' | 'knn' | 'iterative'
    knn_neighbors: int = 10
    n_select_features: int = 1435       # Ridge-RFE on train subjects; 0 = no selection
    graph_metric: str = "pearson"       # pearson | spearman | cosine | euclidean | partial_correlation
    graph_mode: str = "static"          # static | learnable | learnable_scratch
    conv_type: str = "cheb"             # cheb | gcn | graph | gat | gin | sage
    k_neighbors: dict = field(default_factory=lambda: {"aseg": 4, "aparc": 16, "wmparc": 8})
    default_k_neighbors: int = 32
    hidden_dim: int = 128
    cheb_K: int = 5
    dropout: float = 0.5

    # ------------------------------------------------------------------- fMRI
    embedding_size: int = 8             # DHGFormer yaml value

    # ------------------------------------------------- fusion-only (modality='both')
    fusion_dim: int = 32
    fusion_heads: int = 2
    fusion_dropout: float = 0.5
    modality_dropout: float = 0.2
    aux_weight: float = 0.3
