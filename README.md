# ABIDE: sMRI (MultiViewGCN) vs fMRI (DHGFormer) vs Fusion

A single pipeline for three modes — `smri` / `fmri` / `both` — evaluated under **fully identical conditions**.

## Usage

```bash
pip install -r requirements.txt
python main.py --modality smri
python main.py --modality fmri
python main.py --modality both --fusion cross_attention     # or concat
python main.py --modality both --no-mixup                   # mixup off
```

In Colab:

```python
%cd /content/drive/MyDrive/abide_multimodal
from main import run
from config import Config
run(Config(modality='fmri', mixup=False))
```

Every field in `config.py` is also a command-line flag (`--lr 5e-4 --batch_size 32 --epochs 150 ...`).
Set the file paths (`smri_csv`, `pheno_csv`, `fmri_npy`, `fmri_order_txt`, `out_dir`) in `config.py`.

## What is shared across the three modes (for a fair comparison)

| Item | Setting |
|---|---|
| Subjects and folds | Exactly identical; all three modes use only subjects that have **both sMRI and fMRI** (`match_fmri_subjects=True`). Folds are the same split as before (StratifiedKFold within each site, `random_state=0`). |
| Optimizer | Adam with **one** `lr` and **one** `weight_decay` for all parameters (default `1e-4` / `1e-4`). |
| Mini-batch | `batch_size=16` for all three modes (sMRI is no longer full-batch). |
| Epochs / seed | `epochs=100`; the seed of each fold is `seed + fold`. |
| Mixup | A single switch `--mixup/--no-mixup` and a single `mixup_alpha`; in `both` mode it is applied to both modalities with the same `perm` and `lam`. |
| Model selection | Best validation accuracy (ties broken in favor of the later epoch); test results are reported at that same epoch. |
| Gradient clipping | `grad_clip=5`. |
| Main loss weight | `ce_weight=1` for all. |
| sMRI preprocessing | Impute / ComBat / scale / Ridge-RFE / graph construction — all fitted on the training subjects of each fold only. |
| Reported metrics | Train CE (main term only), ACC / AUC / SEN / SPE / F1 for val and test, mean ± std across folds, and pooled. |

## Model-specific components

- `group_loss` and `sparsity_loss` belong to DHGFormer and are applied wherever DHGFormer is used (`fmri` and `both`).
- `aux_weight`, `modality_dropout` and `fusion_*` are used only in `both` mode.
- sMRI-only mode additionally supports `graph_mode='learnable*'` (edge learning rate = `lr * edge_lr_scale`).

## Project structure

```
config.py          all settings
data.py            table loading, fMRI alignment, fold construction
smri_preproc.py    impute / ComBat / scale / RFE / graph (leak-free, train-only fitting)
models/            layers.py, smri.py, fmri.py (DHGFormer), fusion.py
train.py           one training loop for all three modes (optional mixup)
metrics.py         reporting, results.json, plots (PNG)
main.py            entry point
```

Each run writes its output to `out_dir/with_ComBat|without_ComBat/<modality>[_fusion]_<mixup|nomixup>/`:
`config.json`, `results.json`, `best_fold*.pth`, `roc_test.png`, `curves_fold*.png`.
