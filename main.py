"""Entry point.   python main.py --modality smri|fmri|both  [--mixup/--no-mixup] [--lr ...] ...
From a notebook:   from main import run; from config import Config; run(Config(modality='fmri'))"""
import argparse
import dataclasses
import json
import os

import numpy as np
import torch

import data as data_mod
import metrics
import smri_preproc
import train as train_mod
from config import Config
from models import load_cluster_order


def run(cfg):
    assert cfg.modality in ("smri", "fmri", "both"), "modality must be smri / fmri / both"
    assert cfg.fusion in ("concat", "cross_attention")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_s = cfg.modality in ("smri", "both")
    use_f = cfg.modality in ("fmri", "both")
    run_name = cfg.modality + (f"_{cfg.fusion}" if cfg.modality == "both" else "") + ("_mixup" if cfg.mixup else "_nomixup")
    out_dir = os.path.join(cfg.out_dir, "with_ComBat" if cfg.combat else "without_ComBat", run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    print(f"Device: {dev} | modality: {cfg.modality} | mixup: {cfg.mixup} | lr {cfg.lr} | wd {cfg.weight_decay} "
          f"| batch {cfg.batch_size} | epochs {cfg.epochs}\nOutput: {out_dir}\n")

    cohort = data_mod.load_cohort(cfg)
    if use_f or cfg.match_fmri_subjects:
        data_mod.attach_fmri(cfg, cohort)          # also defines the common subject set for sMRI-only runs
    folds = data_mod.make_folds(cohort, cfg.k_fold)
    for f, s in folds.items():
        print(f"  fold {f}: train {len(s['train'])} | val {len(s['val'])} | test {len(s['test'])}")

    layout = combat_mod = None
    if use_s:
        layout = smri_preproc.build_layout(cohort.feature_cols)
        if cfg.combat:
            cat = [c for c in cfg.combat_categorical if c in cohort.pheno.columns]
            cont = [c for c in cfg.combat_continuous if c in cohort.pheno.columns]
            assert "DX_GROUP" not in cat, "DX_GROUP must not be a ComBat covariate (label leakage)"
            combat_mod = smri_preproc.build_combat_covariates(cohort.pheno, cat, cont)
            print(f"ComBat (fitted per fold on TRAIN only) protects: {cat + cont}")
    cluster_order = load_cluster_order(cfg.cluster_map_path) if use_f else None

    results = []
    for fold in range(1, cfg.k_fold + 1):
        split = folds[fold]
        smri_data = smri_preproc.prepare_fold(cohort, cfg, layout, combat_mod, split["train"]) if use_s else None
        results.append(train_mod.run_fold(cfg, cohort, fold, split, smri_data, cluster_order, dev, out_dir))
    return metrics.report(results, cfg, out_dir, run_name)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(Config):
        v = getattr(Config(), f.name)
        if isinstance(v, bool):
            p.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=v)
        elif isinstance(v, (dict, list)):
            continue                                  # edit these in config.py
        else:
            p.add_argument(f"--{f.name}", type=type(v), default=v)
    return p


if __name__ == "__main__":
    ns, _ = build_parser().parse_known_args()
    run(Config(**{k: v for k, v in vars(ns).items()}))
