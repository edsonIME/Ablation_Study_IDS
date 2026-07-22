#!/usr/bin/env python3
"""
generate_synthetic_results.py
=============================
Write synthetic per-run CSVs in the schema produced by ablation_runner.py, so
the analysis pipeline can be exercised without TensorFlow or the real
datasets.

Values are NEUTRAL and obviously artificial (each model's centre is a plain
function of its index), encoding no experimental narrative. The per-seed
partition_hash is shared across models (as real runs share partitions) and the
protocol_hash is shared across models within a dataset -- so the integrity and
cross-model-protocol checks see a valid study.

    python tests/generate_synthetic_results.py --out results_demo
    python statistical_analysis.py --results-dir results_demo
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["M0", "M1", "M2", "M3", "M4", "M5", "M5b", "M6"]
BASES = {"HIKARI": 0.70, "CIRA": 0.60}
# Neutral, obviously-artificial sizes (only the n_test/n_train ratio
# matters for the NB correction); NOT the real cleaned-dataset sizes.
SIZES = {"HIKARI": (7000, 1500), "CIRA": (7000, 1500)}
COLS = ["seed", "run", "n_params", "grid", "tokens", "best_epoch", "t_star",
        "train_time_s", "test_time_s", "n_train", "n_val", "n_test",
        "n_train_smote", "n_features", "frac_train", "frac_val", "frac_test",
        "group_conflicts", "partition_hash", "split_mode", "epochs_max",
        "batch_size", "test_Acc", "test_Prec", "test_Rec", "test_F1",
        "test_AUC", "tstar_Acc", "tstar_Prec", "tstar_Rec", "tstar_F1",
        "tstar_AUC", "model", "dataset", "run_ignored", "dedup_features",
        "max_samples", "dataset_hash", "protocol_hash",
        "study_protocol_hash", "model_hash", "code_version"]


def _h(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description="synthetic runner-format CSVs")
    ap.add_argument("--out", default="results_demo")
    ap.add_argument("--n-runs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    study = _h({"study": "synthetic-shared"})   # SAME for both datasets
    for ds, base in BASES.items():
        n_tr, n_te = SIZES[ds]
        nf = 79 if ds == "HIKARI" else 29
        proto = _h({"ds": ds, "nf": nf})                    # per-dataset
        # partition_hash shared across models for each (ds, seed).
        part = {41 + r: _h({"ds": ds, "seed": 41 + r}) for r in range(1, args.n_runs + 1)}
        for i, model in enumerate(MODELS):
            mu = base + 0.01 * i
            mhash = _h({"model": model})
            rows = []
            for r in range(1, args.n_runs + 1):
                seed = 41 + r
                f1 = float(np.clip(mu + rng.normal(0, 0.01), 0, 1))
                rows.append({
                    "seed": seed, "run": r, "n_params": 500000,
                    "grid": 9 if ds == "HIKARI" else 6,
                    "tokens": 4 if ds == "HIKARI" else 1,
                    "best_epoch": int(rng.integers(5, 60)), "t_star": 0.5,
                    "train_time_s": 1000.0, "test_time_s": 10.0,
                    "n_train": n_tr, "n_val": n_te, "n_test": n_te,
                    "n_train_smote": int(n_tr * 1.1), "n_features": nf,
                    "frac_train": 0.70, "frac_val": 0.15, "frac_test": 0.15,
                    "group_conflicts": 0, "partition_hash": part[seed],
                    "split_mode": "group", "epochs_max": 100, "batch_size": 64,
                    "test_Acc": f1, "test_Prec": f1, "test_Rec": f1,
                    "test_F1": f1, "test_AUC": f1,
                    "tstar_Acc": f1, "tstar_Prec": f1, "tstar_Rec": f1,
                    "tstar_F1": f1, "tstar_AUC": f1,
                    "model": model, "dataset": ds, "run_ignored": r,
                    "dedup_features": 0, "max_samples": 0,
                    "dataset_hash": f"synthetic-{ds}", "protocol_hash": proto,
                    "study_protocol_hash": study,
                    "model_hash": mhash, "code_version": "v7.9"})
            df = pd.DataFrame(rows)
            df = df[[c for c in COLS if c in df.columns]]
            df.to_csv(out / f"{model}_{ds}.csv", index=False)
    print(f"[ok] synthetic results written to {out}/ "
          f"({len(BASES)} datasets x {len(MODELS)} models)")


if __name__ == "__main__":
    main()
