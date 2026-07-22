"""
extended_analysis.py
====================
Auxiliary, deeper statistics complementary to statistical_analysis.py. To
guarantee it NEVER diverges from the canonical analysis, the pairwise layer
prepares and tests every contrast through the canonical ``analyse_contrast``
(same seed intersection, same partition_hash verification -- which ABORTS on a
mismatch -- same per-contrast NB ratio, same test and CIs) over the SAME Holm
family (reference vs every model, plus the placement contrast M5 vs M5b). Both
loaders here run the canonical integrity, cross-file-duplicate and
cross-model-protocol checks.

This module only adds analyses that live nowhere else: convergence-failure
rates and median/IQR; the ECA dose-response slope (dose 1 = per-run mean of
M5 AND M5b, both required; its bootstrap CI is an UNCORRECTED descriptive
interval); a validation-tuned threshold helper; and model-complexity helpers.

Usage:
    python extended_analysis.py --results-dir results
"""

import argparse
import os

import numpy as np
import pandas as pd

from statistical_analysis import (analyse_contrast, holm_adjust, bootstrap_ci,
                                   load_results_dir, load_runner_long)

PRIMARY_METRIC = "f1"
ALPHA = 0.05
FAIL_THRESHOLD = 0.5
EQUIV_MARGIN_PP = 0.5
REFERENCE = "M1"
PLACEMENT_CONTRAST = ("M5", "M5b")
ECA_DOSE_MODELS = {0: ["M4"], 1: ["M5", "M5b"], 2: ["M1"], 3: ["M6"]}


def descriptive_table(df, metric=PRIMARY_METRIC):
    """Mean/std, median/IQR and convergence-failure rate per model/dataset."""
    rows = []
    for (model, dataset), g in df.groupby(["model", "dataset"]):
        v = g[metric].to_numpy()
        q1, q3 = np.percentile(v, [25, 75])
        rows.append({"model": model, "dataset": dataset, "n_runs": len(v),
                     "mean": v.mean(), "std": v.std(ddof=1),
                     "median": np.median(v), "iqr_low": q1, "iqr_high": q3,
                     "failed_runs": int((v < FAIL_THRESHOLD).sum()),
                     "failure_rate": float((v < FAIL_THRESHOLD).mean())})
    return pd.DataFrame(rows)


def paired_comparisons(table, meta, fallback_ratio):
    """Canonical family, prepared and tested by analyse_contrast -- identical
    numbers to statistical_analysis.run_analysis (including the partition_hash
    abort and the per-contrast NB ratio)."""
    fb = fallback_ratio or (0.70, 0.15)
    results = []
    for dataset in sorted(table):
        models = table[dataset]
        if REFERENCE not in models:
            continue
        contrasts = [(REFERENCE, m) for m in sorted(models) if m != REFERENCE]
        if all(m in models for m in PLACEMENT_CONTRAST):
            contrasts.append(PLACEMENT_CONTRAST)
        entries = []
        for a, b in contrasts:
            res, n_seeds = analyse_contrast(table, meta, dataset, a, b,
                                            EQUIV_MARGIN_PP, fb)
            if res is None:
                continue
            entries.append({
                "dataset": dataset, "comparison": f"{a} vs {b}",
                "n_pairs": res["n_pairs"], "test": "NB-corrected paired t",
                "p_raw": res["p_raw"], "effect_type": "cohen_dz",
                "effect": res["effect"], "delta_pp": res["delta_pp"],
                "delta_pp_ci95": (round(res["delta_ci"][0], 2),
                                  round(res["delta_ci"][1], 2)),
                "wilcoxon_p_uncorr": res["wilcoxon_p"],
                "shapiro_p": res["shapiro_p"]})
        for entry, p_adj in zip(entries, holm_adjust([e["p_raw"]
                                                      for e in entries])):
            entry["p_holm"] = p_adj
            entry["significant"] = p_adj < ALPHA
        results.extend(entries)
    return pd.DataFrame(results)


def eca_gradient(df, metric=PRIMARY_METRIC):
    """F1 vs number of ECA modules. Dose 1 = per-run mean of M5 AND M5b (both
    required). NOTE: the bootstrap CI treats runs as independent, so it is an
    UNCORRECTED descriptive interval (unlike the NB-corrected contrasts)."""
    seed = "seed" if "seed" in df.columns else "run"
    results = []
    for dataset, g in df.groupby("dataset"):
        wide = g.pivot_table(index=seed, columns="model", values=metric)
        dose_cols, doses = {}, []
        for dose, models in sorted(ECA_DOSE_MODELS.items()):
            # Require ALL models of the dose to be present as columns (dose 1
            # needs BOTH M5 and M5b) and non-null per run (skipna=False).
            if not all(m in wide.columns for m in models):
                continue
            dose_cols[dose] = wide[models].mean(axis=1, skipna=False)
            doses.append(dose)
        if len(doses) < 2:
            print(f"[gradient] {dataset}: fewer than 2 dose points -- skipped")
            continue
        w = pd.DataFrame({d: dose_cols[d] for d in doses}).dropna()
        if len(w) < 5:
            print(f"[gradient] {dataset}: <5 complete runs across doses -- skipped")
            continue
        x = np.array(doses, dtype=float)
        xc = x - x.mean()
        y = w.to_numpy()
        slopes = (y - y.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)
        v = 100.0 * slopes
        lo, hi = bootstrap_ci(v, np.mean)
        results.append({"dataset": dataset,
                        "doses_used": "+".join(map(str, doses)),
                        "n_runs": len(w),
                        "slope_pp_per_module": v.mean(),
                        "uncorrected_ci95": (round(lo, 2), round(hi, 2))})
    return pd.DataFrame(results)


def tuned_threshold_f1(y_val, s_val, y_test, s_test):
    from sklearn.metrics import precision_recall_curve, f1_score
    prec, rec, thr = precision_recall_curve(y_val, s_val)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    best = thr[np.argmax(f1[:-1])]
    return f1_score(y_test, s_test > best), float(best)  # '>' matches runner


def keras_complexity(model, batch_input_shape):
    import tensorflow as tf
    from tensorflow.python.profiler.model_analyzer import profile
    from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
    n_params = model.count_params()
    forward = tf.function(
        lambda x: model(x),
        input_signature=[tf.TensorSpec([1] + list(batch_input_shape[1:]))])
    flops = profile(forward.get_concrete_function().graph,
                    options=ProfileOptionBuilder.float_operation()).total_float_ops
    return {"params": n_params, "size_mb": round(n_params * 4 / 2**20, 2),
            "flops_per_sample": flops}


def to_latex_rows(stats_df):
    lines = []
    for _, r in stats_df.iterrows():
        ci = r["delta_pp_ci95"]
        lines.append(f"{r['dataset']} & {r['comparison']} & {r['test']} & "
                     f"{r['p_raw']:.2e} & {r['p_holm']:.2e} & "
                     f"{r['delta_pp']:+.2f} [{ci[0]:+.2f}; {ci[1]:+.2f}] & "
                     f"$d_z={r['effect']:+.2f}$ \\\\")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Auxiliary ablation statistics reusing the canonical "
                    "loader and contrast preparation from statistical_analysis.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", default="stats_report")
    parser.add_argument("--n-train", type=int, default=None,
                        help="fallback NB n_train if the CSVs lack the sizes")
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--allow-legacy-results", action="store_true")
    args = parser.parse_args()

    fallback = None
    if args.n_train and args.n_test:
        fallback = (float(args.n_train), float(args.n_test))
    os.makedirs(args.out, exist_ok=True)

    # Long df (all metrics) for descriptive/gradient; table+meta for the paired
    # layer -- both via the canonical loaders (same integrity checks).
    df = load_runner_long(args.results_dir, args.allow_legacy_results)
    table, meta = load_results_dir(args.results_dir, "test_F1",
                                   args.allow_legacy_results)

    desc = descriptive_table(df)
    desc.to_csv(os.path.join(args.out, "descriptive.csv"), index=False)
    print("\n=== Descriptive (incl. convergence failures) ===")
    print(desc.to_string(index=False))

    comp = paired_comparisons(table, meta, fallback)
    comp.to_csv(os.path.join(args.out, "pairwise.csv"), index=False)
    print("\n=== Paired comparisons (canonical analyse_contrast; PRIMARY NB-t, "
          "Holm-adjusted; Wilcoxon = uncorrected sensitivity) ===")
    print(comp.to_string(index=False))

    grad = eca_gradient(df)
    if not grad.empty:
        grad.to_csv(os.path.join(args.out, "eca_gradient.csv"), index=False)
        print("\n=== ECA gradient (slope pp/module; UNCORRECTED bootstrap CI) ===")
        print(grad.to_string(index=False))

    with open(os.path.join(args.out, "table_rows.tex"), "w") as fh:
        fh.write(to_latex_rows(comp))
    print("\n=== LaTeX rows written to table_rows.tex ===")


if __name__ == "__main__":
    main()
