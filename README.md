# ABIDE: sMRI (MultiViewGCN) vs fMRI (DHGFormer) vs fusion

یک پایپ‌لاین واحد برای سه حالت `smri` / `fmri` / `both` با **شرایط کاملاً یکسان**.

## اجرا

```bash
pip install -r requirements.txt
python main.py --modality smri
python main.py --modality fmri
python main.py --modality both --fusion cross_attention     # یا concat
python main.py --modality both --no-mixup                   # mixup خاموش
```

در Colab:

```python
%cd /content/drive/MyDrive/abide_multimodal
from main import run
from config import Config
run(Config(modality='fmri', mixup=False))
```

همه‌ی فیلدهای `config.py` فلگ خط فرمان هم هستند (`--lr 5e-4 --batch_size 32 --epochs 150 ...`).
مسیر فایل‌ها (`smri_csv`, `pheno_csv`, `fmri_npy`, `fmri_order_txt`, `out_dir`) را در `config.py` تنظیم کنید.

تست سریع روی داده‌ی ساختگی (CPU):

```bash
python tools/make_synthetic_data.py /tmp/fake_abide
python main.py --smri_csv /tmp/fake_abide/abide_smri.csv --pheno_csv /tmp/fake_abide/pheno.csv \
  --fmri_npy /tmp/fake_abide/abide.npy --fmri_order_txt /tmp/fake_abide/subject_order.txt \
  --out_dir /tmp/fake_abide/out --epochs 2 --n_select_features 40 --modality both
```

## چه چیزی بین سه حالت مشترک است (برای مقایسه‌ی منصفانه)

| مورد | مقدار |
|---|---|
| subjectها و foldها | دقیقاً یکسان؛ هر سه حالت فقط روی subjectهایی که **هم sMRI و هم fMRI** دارند (`match_fmri_subjects=True`). foldها همان تقسیم قبلی هستند (StratifiedKFold داخل هر سایت، `random_state=0`) |
| optimizer | Adam با **یک** `lr` و **یک** `weight_decay` برای همه‌ی پارامترها (پیش‌فرض `1e-4` / `1e-4`) |
| mini-batch | `batch_size=16` برای هر سه حالت (sMRI هم دیگر full-batch نیست) |
| epochs / seed | `epochs=100`، seed هر fold برابر `seed + fold` |
| mixup | یک سوییچ `--mixup/--no-mixup` و یک `mixup_alpha`؛ در حالت both روی هر دو مدالیتی با یک `perm` و یک `lam` |
| انتخاب مدل | بهترین accuracy روی validation (در تساوی، epoch دیرتر)، گزارش روی test همان epoch |
| gradient clipping | `grad_clip=5` |
| وزن loss اصلی | `ce_weight=1` برای همه |
| پیش‌پردازش sMRI | impute / ComBat / scale / Ridge-RFE / گراف، همه فقط روی train هر fold fit می‌شوند |
| معیار گزارش | train CE (فقط جمله‌ی اصلی)، ACC / AUC / SEN / SPE / F1 برای val و test، میانگین±انحراف معیار و pooled |

## چیزهایی که ذاتاً مخصوص مدل هستند

- `group_loss` و `sparsity_loss` مربوط به DHGFormer هستند و هر جا DHGFormer هست (fmri و both) اعمال می‌شوند.
- `aux_weight`، `modality_dropout`، `fusion_*` فقط در `both` استفاده می‌شوند.
- sMRI-only حالت `graph_mode='learnable*'` را هم دارد (lr لبه‌ها = `lr * edge_lr_scale`).

## ساختار

```
config.py          همه‌ی تنظیمات
data.py            خواندن جدول‌ها، تطبیق fMRI، ساخت foldها
smri_preproc.py    impute/ComBat/scale/RFE/گراف (leak-free، فقط train)
models/            layers.py, smri.py, fmri.py (DHGFormer), fusion.py
train.py           یک حلقه‌ی آموزش برای هر سه حالت (mixup اختیاری)
metrics.py         گزارش، results.json، نمودارها (PNG)
main.py            نقطه‌ی ورود
```

خروجی هر اجرا در `out_dir/with_ComBat|without_ComBat/<modality>[_fusion]_<mixup|nomixup>/`:
`config.json`, `results.json`, `best_fold*.pth`, `roc_test.png`, `curves_fold*.png`.
