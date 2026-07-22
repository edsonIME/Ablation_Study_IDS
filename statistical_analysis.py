#!/usr/bin/env python3
"""
statistical_analysis.py
=======================
Canonical statistical analysis for the CNN-ECA-Transformer ablation study --
the SINGLE source of truth for the paired protocol. The runner
(``--stats-only``) and ``extended_analysis`` both call the functions here, and
in particular both prepare every contrast through ``analyse_contrast``, so
pairing, partition verification, the NB ratio and the test are identical.

Integrity enforced at load time (strict by default)
---------------------------------------------------
* required provenance columns must be present on every runner CSV
  (seed, model, dataset, protocol_hash, model_hash, dataset_hash,
  partition_hash, n_train, n_test); pass ``--allow-legacy-results`` /
  ``allow_legacy=True`` only for older files, accepting the loss of these
  guarantees;
* pairing is by SEED;
* a per-seed ``partition_hash`` must MATCH between the two models of every
  contrast -- a mismatch ABORTS (the paired test would otherwise mix an
  architecture effect with a different resample);
* within each dataset every compared model must share one ``protocol_hash``;
* duplicate (model,dataset,seed) rows are rejected WITHIN a file AND ACROSS
  files (a stray copy can never silently overwrite a result).

Pre-specified test: NB-corrected paired t (PRIMARY, with the n_test/n_train
ratio from the shared seeds' recorded sizes); Wilcoxon is an uncorrected
sensitivity; Shapiro-Wilk is a diagnostic. Nothing is hard-coded.

Reference: Nadeau & Bengio (2003), Machine Learning 52.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TRAIN_FRAC = 0.70
TEST_FRAC = 0.15


class IncompleteResults(SystemExit):
    """Raised when there are not enough completed runs to analyse
    (e.g. the reference model or >=5 shared seeds are missing). This
    is distinct from an integrity violation (bad/blank/inconsistent
    provenance), which raises a plain SystemExit and is never
    tolerated."""
DEFAULT_RESULT_DIRS = ("results", "../results", "./results")
RUN_ID_COLS = ("seed", "run")          # SEED preferred for pairing
DEFAULT_LATEX_FILE = "statistical_tables.tex"
ALL_METRICS = {"test_F1": "f1", "test_Prec": "precision", "test_Rec": "recall",
               "test_Acc": "accuracy", "test_AUC": "auc", "tstar_F1": "f1_tstar"}
# Provenance columns required from runner-produced CSVs (strict mode).
REQUIRED_PROVENANCE = ("seed", "model", "dataset", "protocol_hash",
                       "study_protocol_hash", "model_hash",
                       "dataset_hash", "partition_hash",
                       "n_train", "n_test", "code_version")
# Metric columns that must lie in [0, 1] when present.
RATE_METRICS = ("f1", "auc", "precision", "recall", "accuracy",
                "test_F1", "test_AUC", "test_Prec", "test_Rec", "test_Acc",
                "tstar_F1", "tstar_AUC", "tstar_Prec", "tstar_Rec", "tstar_Acc")
POSITIVE_INT_COLS = ("n_train", "n_test", "n_val", "n_train_smote")

# Expected shape of a COMPLETE study (for --require-complete). These describe
# the declared protocol, not any result.
EXPECTED_MODELS = ("M0", "M1", "M2", "M3", "M4", "M5", "M5b", "M6")
EXPECTED_DATASETS = ("HIKARI", "CIRA")
from config import EXPECTED_SEEDS               # 30 Monte Carlo runs, 42..71


def require_complete_study(table):
    """Abort unless every expected (dataset, model) has all 30 expected seeds.
    Use for the FINAL tables so a campaign interrupted early cannot masquerade
    as complete."""
    problems = []
    for ds in EXPECTED_DATASETS:
        for m in EXPECTED_MODELS:
            seeds = set(table.get(ds, {}).get(m, {}))
            missing = [s for s in EXPECTED_SEEDS if s not in seeds]
            extra = sorted(seeds - set(EXPECTED_SEEDS))
            if missing:
                problems.append(f"{ds}/{m}: missing {len(missing)} seed(s) "
                                f"(e.g. {missing[:3]})")
            if extra:
                problems.append(f"{ds}/{m}: unexpected seed(s) {extra[:3]}")
    if problems:
        raise SystemExit("Study is INCOMPLETE (--require-complete):\n  " +
                         "\n  ".join(problems))
    print(f"[complete] all {len(EXPECTED_DATASETS)}x{len(EXPECTED_MODELS)} "
          f"model-dataset pairs have {len(EXPECTED_SEEDS)} seeds.")

# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def nb_se(sd_diff, n, train_size, test_size):
    return sd_diff * np.sqrt(1.0 / n + test_size / train_size)


def nb_corrected_ttest(diff, train_size, test_size):
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    se = nb_se(diff.std(ddof=1), n, train_size, test_size)
    if se == 0.0:
        return np.inf * np.sign(diff.mean()), 0.0 if diff.mean() != 0 else 1.0
    t_stat = diff.mean() / se
    return t_stat, 2.0 * stats.t.sf(abs(t_stat), df=n - 1)


def nb_mean_diff_ci(diff, train_size, test_size, level=0.95):
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    se = nb_se(diff.std(ddof=1), n, train_size, test_size)
    half = stats.t.ppf(0.5 + level / 2, df=n - 1) * se
    return diff.mean() - half, diff.mean() + half


def cohens_dz(diff):
    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    return np.inf * np.sign(diff.mean()) if sd == 0 else diff.mean() / sd


def rank_biserial(diff):
    d = np.asarray(diff, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return (ranks[d > 0].sum() - ranks[d < 0].sum()) / (ranks.sum())


def bootstrap_ci(values, stat_fn, n_boot=10_000, alpha=0.05, seed=42):
    """Percentile bootstrap CI; degenerate inputs return a point interval
    instead of crashing on an empty percentile."""
    values = np.asarray(values, dtype=float)
    full = stat_fn(values)
    if np.allclose(values, values[0]):
        return (float(full), float(full)) if np.isfinite(full) else (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = stat_fn(values[rng.integers(0, n, n)])
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return (float(full), float(full)) if np.isfinite(full) else (np.nan, np.nan)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted


def compare_pair(f1_a, f1_b, train_size, test_size, equiv_margin_pp):
    """Canonical paired protocol for ONE contrast. Handles constant
    differences without crashing (no bootstrap on a degenerate sample)."""
    diff = np.asarray(f1_a, dtype=float) - np.asarray(f1_b, dtype=float)
    n = len(diff)
    delta_pp = 100.0 * diff.mean()
    constant = np.allclose(diff, diff[0])

    if constant:
        c = float(diff[0])
        shapiro_p = float("nan")
        if c == 0.0:
            p_nb, dz, p_w, r_prb = 1.0, 0.0, 1.0, 0.0
            d_lo = d_hi = 0.0
            dz_lo = dz_hi = 0.0
        else:
            p_nb = 0.0
            dz = np.inf * np.sign(c)
            p_w = 2.0 ** (1 - n)
            r_prb = float(np.sign(c))
            d_lo = d_hi = delta_pp
            dz_lo = dz_hi = dz
    else:
        shapiro_p = float(stats.shapiro(diff).pvalue)
        _, p_nb = nb_corrected_ttest(diff, train_size, test_size)
        dz = cohens_dz(diff)
        dz_lo, dz_hi = bootstrap_ci(diff, cohens_dz)
        d_lo, d_hi = nb_mean_diff_ci(diff, train_size, test_size)
        d_lo, d_hi = 100.0 * d_lo, 100.0 * d_hi
        try:
            p_w = float(stats.wilcoxon(diff).pvalue)
        except ValueError:
            p_w = 1.0
        r_prb = rank_biserial(diff)

    equivalent = np.isfinite(d_lo) and np.isfinite(d_hi) \
        and (d_lo > -equiv_margin_pp) and (d_hi < equiv_margin_pp)
    return {"test": "NB-t", "p_raw": p_nb, "delta_pp": delta_pp,
            "delta_ci": (d_lo, d_hi), "effect": dz, "effect_ci": (dz_lo, dz_hi),
            "equivalent": equivalent, "n_pairs": n, "shapiro_p": shapiro_p,
            "wilcoxon_p": p_w, "rank_biserial": r_prb}


def paired_contrast(table, meta, dataset, a, b):
    """Shared contrast preparation used by BOTH run_analysis and
    extended_analysis, so they cannot diverge. Pairs by seed, ABORTS on a
    partition_hash mismatch, and returns the NB (train_size, test_size) from
    the shared seeds' recorded sizes (None if unavailable)."""
    seeds = sorted(set(table[dataset][a]) & set(table[dataset][b]))
    dmeta = meta.get(dataset, {})
    part = dmeta.get("partition", {})
    mism = [s for s in seeds if (a, s) in part and (b, s) in part
            and part[(a, s)] != part[(b, s)]]
    if mism:
        raise SystemExit(
            f"{dataset} {a} vs {b}: partition_hash MISMATCH for "
            f"{len(mism)} shared seed(s) {mism[:5]}{'...' if len(mism) > 5 else ''}. "
            f"The two models were not trained on identical partitions, so a "
            f"paired test is invalid. Re-run so both use the same partitions.")
    sizes = dmeta.get("sizes", {})           # keyed by (model, seed)
    tr, te, size_mism = [], [], []
    for sd in seeds:
        sa_, sb_ = sizes.get((a, sd)), sizes.get((b, sd))
        if sa_ is not None and sb_ is not None and sa_ != sb_:
            size_mism.append(sd)
        if sa_ is not None:
            tr.append(sa_[0]); te.append(sa_[1])
    if size_mism:
        raise SystemExit(
            f"{dataset} {a} vs {b}: n_train/n_test DIFFER between the two "
            f"models for shared seed(s) {size_mism[:5]}"
            f"{'...' if len(size_mism) > 5 else ''}. The paired runs must use "
            f"the same partition sizes; the CSVs are inconsistent.")
    ratio = (float(np.mean(tr)), float(np.mean(te))) if tr and te else None
    a_vals = [table[dataset][a][s] for s in seeds]
    b_vals = [table[dataset][b][s] for s in seeds]
    return seeds, a_vals, b_vals, ratio


def analyse_contrast(table, meta, dataset, a, b, equiv_margin_pp,
                     fallback_ratio):
    """paired_contrast + compare_pair. Returns (result_dict, n_seeds)."""
    seeds, a_vals, b_vals, ratio = paired_contrast(table, meta, dataset, a, b)
    if len(seeds) < 5:
        return None, len(seeds)
    tr, te = ratio or fallback_ratio
    return compare_pair(a_vals, b_vals, tr, te, equiv_margin_pp), len(seeds)

# ---------------------------------------------------------------------------
# LaTeX table composers
# ---------------------------------------------------------------------------

DOSE_LABELS = {"M4": "M4 (0)", "M5": "M5 (1, b1)", "M5b": "M5b (1, b2)",
               "M1": "M1 (2)", "M6": "M6 (3)"}
DOSE_ORDER = ["M4", "M5", "M5b", "M1", "M6"]
METRIC_LABEL = {"test_F1": "F1", "tstar_F1": "F1", "test_AUC": "AUC",
                "test_Prec": "Precision", "test_Rec": "Recall",
                "test_Acc": "Accuracy"}


def _mlabel(metric):
    return METRIC_LABEL.get(metric, metric)


def fmt_p_latex(p):
    if p >= 0.01:
        return f"{p:.3f}"
    mant, exp = f"{p:.2e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def compose_stats_table(collected, banner, effect_header, equiv_margin_pp=0.5,
                        metric_label="F1", label="tab:stats"):
    lines = [banner,
             "\\begin{table}[t]\\centering\\scriptsize",
             "\\setlength{\\tabcolsep}{3pt}",
             f"\\caption{{Paired contrasts on \\textbf{{test-set}} {metric_label} "
             "(NB-corrected paired $t$, PRIMARY; $p$ Holm-adjusted per "
             "dataset; Wilcoxon reported as uncorrected sensitivity in the "
             "log). $\\Delta$ $=$ first minus second model, in percentage "
             f"points. $^{{\\dag}}$ $=$ practically equivalent at "
             f"$\\pm{equiv_margin_pp:g}$~pp (CI-based criterion).}}",
             f"\\label{{{label}}}",
             "\\begin{tabular}{@{}llcccc@{}}",
             "\\toprule",
             "\\textbf{Dataset} & \\textbf{Contrast} & \\textbf{Test} & "
             f"\\textbf{{$p$ (adj.)}} & "
             f"\\textbf{{$\\Delta${metric_label} pp [95\\% CI]}} & "
             f"\\textbf{{{effect_header}}} \\\\",
             "\\midrule"]
    prev_ds = None
    for ds, contrast, test, p_adj, delta_cell, effect_cell, equiv in collected:
        if prev_ds is not None and ds != prev_ds:
            lines.append("\\midrule")
        prev_ds = ds
        dag = "$^{\\dag}$" if equiv else ""
        lines.append(f"{ds} & {contrast}{dag} & {test} & "
                     f"{fmt_p_latex(p_adj)} & {delta_cell} & {effect_cell} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def compose_dose_table(table, metric):
    ml = _mlabel(metric)
    datasets = [ds for ds in ("HIKARI", "CIRA") if ds in table]
    datasets += [ds for ds in sorted(table) if ds not in datasets]
    header_cells = " & ".join(f"\\textbf{{{ml} {ds}}}" for ds in datasets)
    lines = ["% -------- ECA dose-response (means computed from data) --------",
             "\\begin{table}[t]\\centering\\footnotesize",
             "\\caption{Channel-attention dose--response "
             f"(\\textbf{{test set}} {ml}; mean~$\\pm$~std over the runs in "
             "\\texttt{results/}).}",
             "\\label{tab:dose}",
             "\\begin{tabular}{@{}l" + "c" * len(datasets) + "@{}}",
             "\\toprule",
             f"\\textbf{{Model (\\#ECA)}} & {header_cells} \\\\", "\\midrule"]
    for m in DOSE_ORDER:
        cells = []
        for ds in datasets:
            runs = table.get(ds, {}).get(m)
            if runs:
                vals = np.asarray(list(runs.values()), dtype=float)
                cells.append(f"${vals.mean():.4f}"
                             f"{{\\scriptstyle\\pm{vals.std(ddof=1):.4f}}}$")
            else:
                cells.append("--")
        lines.append(f"{DOSE_LABELS[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def save_latex_file(path, banner, collected, effect_header, dose_table=None,
                    equiv_margin_pp=0.5, metric_label="F1"):
    content = compose_stats_table(collected, banner, effect_header,
                                  equiv_margin_pp=equiv_margin_pp,
                                  metric_label=metric_label)
    if dose_table:
        content += "\n" + dose_table
    with open(path, "w") as fh:
        fh.write(content)
    print(f"\n[saved] LaTeX tables written to: {path}")

# ---------------------------------------------------------------------------
# Analysis driver
# ---------------------------------------------------------------------------

def run_analysis(table, reference, extra_contrasts, train_frac, test_frac,
                 equiv_margin_pp, metric, emit_latex, save_path=None,
                 meta=None):
    meta = meta or {}
    fallback = (train_frac, test_frac)
    collected = []
    any_contrast = False
    for dataset in sorted(table):
        models = table[dataset]
        if reference not in models:
            print(f"[{dataset}] reference '{reference}' missing "
                  f"(present: {', '.join(sorted(models)) or 'none'}); skipped.")
            continue
        contrasts = [(reference, m) for m in sorted(models) if m != reference]
        contrasts += [c for c in extra_contrasts
                      if c[0] in models and c[1] in models]
        results, labels = [], []
        for a, b in contrasts:
            res, n_seeds = analyse_contrast(table, meta, dataset, a, b,
                                            equiv_margin_pp, fallback)
            if res is None:
                print(f"[{dataset}] {a} vs {b}: only {n_seeds} shared "
                      f"seed(s) (<5); skipped.")
                continue
            results.append(res)
            labels.append(f"{a} vs {b}")
        if not results:
            print(f"[{dataset}] no contrast had >= 5 shared seeds; skipped.")
            continue
        any_contrast = True
        adj = holm_adjust([r["p_raw"] for r in results])
        for r, p_adj in zip(results, adj):
            r["p_adj"] = p_adj
        for lab, r in zip(labels, results):
            d_lo, d_hi = r["delta_ci"]
            e_lo, e_hi = r["effect_ci"]
            collected.append((
                dataset, lab.replace(" vs ", " vs.\\ "), r["test"], r["p_adj"],
                f"${r['delta_pp']:+.2f}$ [{d_lo:+.2f}, {d_hi:+.2f}]",
                f"${r['effect']:+.2f}$ [{e_lo:+.2f}, {e_hi:+.2f}]",
                r["equivalent"]))
        ratio_note = ("from data" if dataset in meta and meta[dataset].get("sizes")
                      else f"nominal {test_frac}/{train_frac}")
        print(f"\n=== {dataset}  (metric={metric}; PRIMARY=NB-t; NB ratio "
              f"{ratio_note}; Holm family = {len(results)}) ===")
        _dh = "d" + _mlabel(metric) + " pp [95% CI]"
        print(f"{'contrast':<12}{'p_adj(NB)':<12}{_dh:<26}"
              f"{'d_z [95% CI]':<24}{'W p(uncorr)':<12}{'equiv'}")
        for lab, r in zip(labels, results):
            d_lo, d_hi = r["delta_ci"]
            e_lo, e_hi = r["effect_ci"]
            print(f"{lab:<12}{r['p_adj']:<12.3e}"
                  f"{r['delta_pp']:+7.2f} [{d_lo:+.2f}, {d_hi:+.2f}]    "
                  f"{r['effect']:+7.2f} [{e_lo:+.2f}, {e_hi:+.2f}]   "
                  f"{r['wilcoxon_p']:<12.2e}{'YES' if r['equivalent'] else 'no'}")
        if emit_latex:
            print(f"\n% LaTeX rows for tab:stats -- {dataset}")
            for lab, r in zip(labels, results):
                d_lo, d_hi = r["delta_ci"]; e_lo, e_hi = r["effect_ci"]
                a, b = lab.split(" vs ")
                print(f"{dataset} & {a} vs.\\ {b} & {r['test']} & "
                      f"{r['p_adj']:.2e} & ${r['delta_pp']:+.2f}$ "
                      f"[{d_lo:+.2f}, {d_hi:+.2f}] & ${r['effect']:+.2f}$ "
                      f"[{e_lo:+.2f}, {e_hi:+.2f}] \\\\")

    if not any_contrast:
        raise IncompleteResults(
            "No dataset produced a valid contrast. Check that the results "
            "directory contains the reference model and at least one other "
            "model with >= 5 shared seeds.")

    if save_path and collected:
        import datetime
        banner = (
            "% ==============================================================\n"
            f"% Auto-generated by statistical_analysis.py on "
            f"{datetime.date.today()}\n"
            f"% PRIMARY test: NB-corrected paired t; metric = {metric}.\n"
            "% Wilcoxon (uncorrected) is a sensitivity check in the log.\n"
            "% All values computed from results/*.csv -- nothing hard-coded.\n"
            "% Requires \\usepackage{booktabs}.\n"
            "% ==============================================================")
        save_latex_file(save_path, banner, collected,
                        "$d_z$ [95\\% CI, uncorr.]",
                        dose_table=compose_dose_table(table, metric),
                        equiv_margin_pp=equiv_margin_pp,
                        metric_label=_mlabel(metric))

# ---------------------------------------------------------------------------
# I/O and integrity checks
# ---------------------------------------------------------------------------

def _safe_read_csv(path):
    """Read a CSV, turning pandas' empty/parse errors into a clear message."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise SystemExit(f"{path}: CSV file is empty or truncated.")
    except pd.errors.ParserError as exc:
        raise SystemExit(f"{path}: malformed CSV: {exc}")


def _locate_results_dir(explicit):
    tried = []
    candidates = [explicit] if explicit else list(DEFAULT_RESULT_DIRS)
    here = Path(__file__).resolve().parent
    if not explicit:
        candidates += [str(here / d) for d in DEFAULT_RESULT_DIRS]
    for cand in candidates:
        p = Path(cand)
        tried.append(str(p))
        if p.is_dir() and any(p.glob("*.csv")):
            return p
    raise SystemExit(
        "Could not find a results directory with CSV files.\n"
        "Tried: " + ", ".join(tried) + "\n"
        "Generate it first, e.g.: python ablation_runner.py --models M1 M4\n"
        "or point at an existing one with --results-dir PATH.")


def _metric_series(df, path, metric):
    if metric in df.columns:
        return df[metric]
    aliases = {"test_F1": ("f1", "F1"), "tstar_F1": ("tstar_f1", "f1_tstar")}
    for alt in aliases.get(metric, ()):
        if alt in df.columns:
            return df[alt]
    raise IncompleteResults(f"Metric column '{metric}' not found in "
                            f"{path}. Available: "
                            f"{', '.join(map(str, df.columns))}")


def _run_ids(df, path):
    for col in RUN_ID_COLS:
        if col in df.columns:
            raw = pd.to_numeric(df[col], errors="coerce")
            if raw.isna().any():
                raise SystemExit(f"{path}: '{col}' has non-numeric values.")
            if not np.allclose(raw.to_numpy(), np.round(raw.to_numpy())):
                raise SystemExit(f"{path}: '{col}' must be integers "
                                 f"(found fractional values).")
            if (raw < 0).any():
                raise SystemExit(f"{path}: '{col}' must be non-negative.")
            return raw.astype(int)
    raise SystemExit(f"No run identifier in {path}; expected one of "
                     f"{', '.join(RUN_ID_COLS)}.")


def _check_int_col(df, path, col):
    """Return an integer Series for `col`, or abort if non-numeric, fractional
    or negative."""
    raw = pd.to_numeric(df[col], errors="coerce")
    if raw.isna().any():
        raise SystemExit(f"{path}: '{col}' has non-numeric/missing values.")
    if not np.allclose(raw.to_numpy(), np.round(raw.to_numpy())):
        raise SystemExit(f"{path}: '{col}' must be integers (found fractional).")
    if (raw < 0).any():
        raise SystemExit(f"{path}: '{col}' must be non-negative.")
    return raw.astype(int)


def validate_runner_results_file(df, path):
    """THE single definition of a valid runner results file, shared by the
    runner (before resuming) and by the analysis loaders, so there are never
    two notions of "valid". Strict. Validates: every required column present,
    non-null and non-blank (including the completeness columns
    run/n_val/test_F1/tstar_F1/partition_hash); integer non-negative seed/run;
    positive integer sizes; all present rate metrics finite and in [0, 1]; one
    dataset_hash/protocol_hash per dataset and one model_hash per model; no
    duplicate (model,dataset,seed); and no partition_hash reused by two seeds
    within a (dataset,model)."""
    required = list(REQUIRED_PROVENANCE) + ["run", "n_val", "test_F1", "tstar_F1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing required column(s): {sorted(missing)}. "
                         f"Not a complete current-runner file (use "
                         f"--allow-legacy-results to bypass, losing guarantees).")
    for c in required:
        # Version-robust missing/blank detection: coerce to nullable string so
        # NaN stays <NA> across pandas versions, then catch NaN, empty, or
        # whitespace-only values (and the "nan"/"none" literals a NaN cast can
        # produce). Does NOT depend on the column dtype.
        col = df[c].astype("string")
        stripped = col.str.strip()
        bad = (col.isna() | stripped.isna() | stripped.eq("")
               | stripped.str.lower().isin(["nan", "none", "<na>"]))
        if bad.any():
            raise SystemExit(f"{path}: column '{c}' has missing/blank values.")
    _check_int_col(df, path, "seed")
    _check_int_col(df, path, "run")
    for c in POSITIVE_INT_COLS:
        if c in df.columns:
            if (_check_int_col(df, path, c) <= 0).any():
                raise SystemExit(f"{path}: '{c}' must be positive.")
    for m in {mm for mm in RATE_METRICS if mm in df.columns}:
        col = pd.to_numeric(df[m], errors="coerce")
        if col.isna().any() or not np.isfinite(col).all():
            raise SystemExit(f"{path}: metric '{m}' has missing/non-finite values.")
        if not col.between(0.0, 1.0).all():
            raise SystemExit(f"{path}: metric '{m}' outside [0, 1] "
                             f"(min {col.min():.4f}, max {col.max():.4f}).")
    # run must track seed with a constant offset (catches run=999 for one row).
    if (df["seed"].astype(int) - df["run"].astype(int)).nunique() != 1:
        raise SystemExit(f"{path}: 'run' is not consistent with 'seed' "
                         f"(seed - run must be constant).")
    # calibrated threshold within the search grid.
    if "t_star" in df.columns:
        ts = pd.to_numeric(df["t_star"], errors="coerce")
        if ts.isna().any() or not ts.between(0.01, 0.99).all():
            raise SystemExit(f"{path}: 't_star' must be within [0.01, 0.99].")
    # recorded split fractions in [0,1], summing to ~1, consistent with sizes.
    fcols = ("frac_train", "frac_val", "frac_test")
    if all(c in df.columns for c in fcols):
        fr = df[list(fcols)].apply(pd.to_numeric, errors="coerce")
        if fr.isna().any().any() or ((fr < 0) | (fr > 1)).any().any():
            raise SystemExit(f"{path}: split fractions must lie in [0, 1].")
        if not np.allclose(fr.sum(axis=1).to_numpy(), 1.0, atol=0.02):
            raise SystemExit(f"{path}: split fractions must sum to ~1.")
        if all(c in df.columns for c in ("n_train", "n_val", "n_test")):
            total = (df["n_train"] + df["n_val"] + df["n_test"]).astype(float)
            for frac_c, n_c in zip(fcols, ("n_train", "n_val", "n_test")):
                if not np.allclose(fr[frac_c].to_numpy(),
                                   (df[n_c] / total).to_numpy(), atol=0.02):
                    raise SystemExit(f"{path}: '{frac_c}' inconsistent with "
                                     f"'{n_c}'/total.")
    for ds, g in df.groupby("dataset"):
        for col in ("dataset_hash", "protocol_hash", "code_version"):
            u = sorted(map(str, g[col].unique()))
            if len(u) > 1:
                raise SystemExit(f"{path}: dataset {ds} has {len(u)} distinct "
                                 f"{col} ({u}); must be exactly one.")
        for mdl, gm in g.groupby("model"):
            u = sorted(map(str, gm["model_hash"].unique()))
            if len(u) > 1:
                raise SystemExit(f"{path}: {ds}/{mdl} has {len(u)} distinct "
                                 f"model_hash ({u}).")
            ph_seeds = defaultdict(set)
            for _, row in gm[["seed", "partition_hash"]].iterrows():
                ph_seeds[str(row["partition_hash"])].add(int(row["seed"]))
            reused = {ph: sorted(sd) for ph, sd in ph_seeds.items() if len(sd) > 1}
            if reused:
                raise SystemExit(f"{path}: {ds}/{mdl} reuses a partition_hash "
                                 f"across seeds {list(reused.items())[:2]}.")


def _check_file_integrity(df, path, metric, allow_legacy):
    """Per-file checks. In strict mode (default) this validates not only the
    PRESENCE of the provenance columns but their CONTENT: no missing values,
    one dataset_hash/protocol_hash per dataset, one model_hash per
    (dataset,model), positive integer partition sizes, and a finite in-range
    metric. Also: no duplicate (model,dataset,seed) rows."""
    id_col = next((c for c in RUN_ID_COLS if c in df.columns), None)
    if id_col and "dataset" in df.columns:
        key_cols = (["model", "dataset", id_col] if "model" in df.columns
                    else ["dataset", id_col])
        dup = df.duplicated(subset=key_cols)
        if dup.any():
            bad = df.loc[dup, key_cols].drop_duplicates().to_dict("records")
            raise SystemExit(f"{path}: duplicate {tuple(key_cols)} rows: {bad}")

    if allow_legacy:
        return
    validate_runner_results_file(df, path)   # the single canonical validator


def _enforce_cross_model_protocol(proto_by_ds):
    study = sorted(proto_by_ds.pop("__STUDY__", set()))
    if len(study) > 1:
        raise SystemExit(
            f"The datasets were produced under DIFFERENT study-wide "
            f"configurations (study_protocol_hash = {', '.join(study)}). "
            f"HIKARI and CIRA must share one general protocol (epochs, batch, "
            f"split, hyper-parameters, code and library versions).")
    codev = sorted(proto_by_ds.pop("__CODEV__", set()))
    if len(codev) > 1:
        raise SystemExit(
            f"Results mix DIFFERENT code_version values across files "
            f"({', '.join(codev)}); the study must come from one code version.")
    for ds, protos in proto_by_ds.items():
        distinct = sorted(protos)
        if len(distinct) > 1:
            raise SystemExit(
                f"Dataset {ds} compares models trained under DIFFERENT "
                f"protocols (protocol_hash = {', '.join(distinct)}). Refusing "
                f"to compare incompatible runs.")


def _ingest(df, dataset_of, model_of, path, metric, table, meta_acc,
            proto_by_ds, seen):
    values = _metric_series(df, path, metric)
    runs = _run_ids(df, path)
    for i in range(len(df)):
        ds, mdl = str(dataset_of.iloc[i]), str(model_of.iloc[i])
        s, val = int(runs.iloc[i]), values.iloc[i]
        key = (ds, mdl, s)
        if key in seen:
            raise SystemExit(
                f"Duplicate result across files for {key} (second occurrence "
                f"in {path}). A (dataset,model,seed) must appear once in the "
                f"whole results set; remove the stray/backup file.")
        seen.add(key)
        if pd.notna(val):
            table[ds][mdl][s] = float(val)
        if "n_train" in df.columns and "n_test" in df.columns \
                and pd.notna(df["n_train"].iloc[i]):
            meta_acc[ds]["sizes"][(mdl, s)] = (float(df["n_train"].iloc[i]),
                                               float(df["n_test"].iloc[i]))
        if "partition_hash" in df.columns:
            meta_acc[ds]["partition"][(mdl, s)] = str(df["partition_hash"].iloc[i])
        if "protocol_hash" in df.columns and pd.notna(df["protocol_hash"].iloc[i]):
            proto_by_ds[ds].add(str(df["protocol_hash"].iloc[i]))
        if "study_protocol_hash" in df.columns and \
                pd.notna(df["study_protocol_hash"].iloc[i]):
            proto_by_ds["__STUDY__"].add(str(df["study_protocol_hash"].iloc[i]))
        if "code_version" in df.columns and pd.notna(df["code_version"].iloc[i]):
            proto_by_ds["__CODEV__"].add(str(df["code_version"].iloc[i]))
        if "dataset_hash" in df.columns and pd.notna(df["dataset_hash"].iloc[i]):
            meta_acc[ds]["dhash"].add(str(df["dataset_hash"].iloc[i]))
        if "model_hash" in df.columns and pd.notna(df["model_hash"].iloc[i]):
            meta_acc[ds]["mhash"][mdl].add(str(df["model_hash"].iloc[i]))


def _finalize(table, meta_acc, proto_by_ds):
    _enforce_cross_model_protocol(proto_by_ds)
    for ds, d in meta_acc.items():
        if len(d.get("dhash", set())) > 1:
            raise SystemExit(
                f"Dataset {ds} has {len(d['dhash'])} distinct dataset_hash "
                f"across files ({sorted(d['dhash'])}); the models were not "
                f"trained on the same data.")
        # One model_hash per (dataset, model), ACROSS files.
        for mdl, hs in d.get("mhash", {}).items():
            if len(hs) > 1:
                raise SystemExit(
                    f"{ds}/{mdl} has {len(hs)} distinct model_hash across "
                    f"files ({sorted(hs)}); the same model was produced by "
                    f"different architectures/code.")
        # Per (dataset, seed): all models must share the partition_hash.
        by_seed = defaultdict(set)
        for (mdl, sd), ph in d.get("partition", {}).items():
            by_seed[sd].add(ph)
        bad = [sd for sd, hs in by_seed.items() if len(hs) > 1]
        if bad:
            raise SystemExit(
                f"Dataset {ds}: seed(s) {sorted(bad)[:5]} have DIFFERENT "
                f"partition_hash across models; runs are not on identical "
                f"partitions.")
        # No partition_hash may be REUSED by different seeds (that would mean
        # the "Monte Carlo" runs are not distinct resamples).
        ph_to_seeds = defaultdict(set)
        for (mdl, sd), ph in d.get("partition", {}).items():
            ph_to_seeds[ph].add(sd)
        reused = {ph: sorted(sds) for ph, sds in ph_to_seeds.items()
                  if len(sds) > 1}
        if reused:
            some = list(reused.items())[:3]
            raise SystemExit(
                f"Dataset {ds}: identical partition_hash reused by DIFFERENT "
                f"seeds {some}. The runs are not distinct Monte Carlo "
                f"resamples (they must be independently sampled partitions).")
    meta = {ds: {"sizes": d["sizes"], "partition": d["partition"]}
            for ds, d in meta_acc.items() if d["sizes"] or d["partition"]}
    return table, meta


def load_results_dir(results_dir, metric="test_F1", allow_legacy=False):
    """Load per-(model,dataset) CSVs. Returns (table, meta). Rejects missing
    provenance and cross-file duplicates unless allow_legacy=True."""
    table = defaultdict(lambda: defaultdict(dict))
    meta_acc = defaultdict(lambda: {"sizes": {}, "partition": {}, "dhash": set(), "mhash": defaultdict(set)})
    proto_by_ds = defaultdict(set)
    seen = set()
    n_files = 0
    for path in sorted(Path(results_dir).glob("*.csv")):
        df = _safe_read_csv(path)
        if df.empty:
            print(f"[warn] {path} is empty; skipped."); continue
        _check_file_integrity(df, path, metric, allow_legacy)
        stem = path.stem.split("_", 1)
        model_of = (df["model"].astype(str) if "model" in df.columns
                    else pd.Series([stem[0]] * len(df)))
        dataset_of = (df["dataset"].astype(str) if "dataset" in df.columns
                      else pd.Series([stem[1] if len(stem) > 1 else "UNKNOWN"]
                                     * len(df)))
        _ingest(df, dataset_of, model_of, path, metric, table, meta_acc,
                proto_by_ds, seen)
        n_files += 1
    if n_files == 0:
        raise SystemExit(f"No usable CSV files in {results_dir!r}.")
    table, meta = _finalize(table, meta_acc, proto_by_ds)
    print(f"[load] {n_files} CSV file(s) from {results_dir} | metric='{metric}'"
          f" | datasets={sorted(table)} | NB ratio from data for: "
          f"{sorted(ds for ds in meta if meta[ds]['sizes']) or 'none'}"
          + ("  [LEGACY MODE: integrity guarantees relaxed]" if allow_legacy else ""))
    return table, meta


def load_long_csv(path, metric="test_F1", allow_legacy=False):
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"CSV file not found: {path}")
    df = _safe_read_csv(p)
    for req in ("model", "dataset"):
        if req not in df.columns:
            raise SystemExit(f"Column '{req}' missing in {path}.")
    _check_file_integrity(df, path, metric, allow_legacy)
    table = defaultdict(lambda: defaultdict(dict))
    meta_acc = defaultdict(lambda: {"sizes": {}, "partition": {}, "dhash": set(), "mhash": defaultdict(set)})
    proto_by_ds = defaultdict(set)
    _ingest(df, df["dataset"].astype(str), df["model"].astype(str), path,
            metric, table, meta_acc, proto_by_ds, set())
    if not table:
        raise SystemExit(f"No usable rows parsed from {path}.")
    table, meta = _finalize(table, meta_acc, proto_by_ds)
    print(f"[load] long-format CSV {path} | metric='{metric}' | "
          f"datasets={sorted(table)}")
    return table, meta


def load_runner_long(results_dir, allow_legacy=False):
    """Shared long-format loader for extended_analysis: one row per run with
    all metrics, seed, sizes and hashes. Runs the SAME integrity, cross-file
    duplicate and cross-model-protocol checks as the canonical loader."""
    frames, proto_by_ds, seen = [], defaultdict(set), set()
    for path in sorted(Path(results_dir).glob("*.csv")):
        df = _safe_read_csv(path)
        if df.empty:
            continue
        _check_file_integrity(df, path, "test_F1", allow_legacy)
        stem = path.stem.split("_", 1)
        model = df["model"].astype(str) if "model" in df.columns \
            else pd.Series([stem[0]] * len(df))
        dataset = df["dataset"].astype(str) if "dataset" in df.columns \
            else pd.Series([stem[1] if len(stem) > 1 else "UNKNOWN"] * len(df))
        id_col = next((c for c in RUN_ID_COLS if c in df.columns), None)
        for i in range(len(df)):
            key = (str(dataset.iloc[i]), str(model.iloc[i]),
                   int(df[id_col].iloc[i]) if id_col else i)
            if key in seen:
                raise SystemExit(f"Duplicate result across files for {key} "
                                 f"(second occurrence in {path}).")
            seen.add(key)
        cols = {"model": model, "dataset": dataset}
        for col in RUN_ID_COLS:
            if col in df.columns:
                cols[col] = df[col]
        for src, dst in ALL_METRICS.items():
            if src in df.columns:
                cols[dst] = df[src]
        for opt in ("n_train", "n_test", "n_val", "partition_hash",
                    "protocol_hash", "study_protocol_hash",
                    "model_hash", "dataset_hash", "code_version"):
            if opt in df.columns:
                cols[opt] = df[opt]
        frames.append(pd.DataFrame(cols))
        if "protocol_hash" in df.columns:
            for ds in dataset.unique():
                sub = df[dataset.values == ds]
                proto_by_ds[str(ds)].update(
                    map(str, sub["protocol_hash"].dropna().unique()))
    if not frames:
        raise SystemExit(f"no CSV files found in {results_dir!r}")
    _enforce_cross_model_protocol(proto_by_ds)
    frame = pd.concat(frames, ignore_index=True)
    for col, what in (("study_protocol_hash", "study-wide configurations"),
                      ("code_version", "code versions")):
        if col in frame.columns:
            u = sorted(map(str, frame[col].dropna().unique()))
            if len(u) > 1:
                raise SystemExit(f"Results mix DIFFERENT {what} ({u}).")
    # Same global invariants the canonical loader enforces.
    if "protocol_hash" in frame.columns and "model_hash" in frame.columns:
        seed_col = next((c for c in RUN_ID_COLS if c in frame.columns), None)
        for ds, g in frame.groupby("dataset"):
            if "dataset_hash" in g.columns:
                dh = sorted(map(str, g["dataset_hash"].dropna().unique()))
                if len(dh) > 1:
                    raise SystemExit(f"{ds}: {len(dh)} distinct dataset_hash "
                                     f"across files ({dh}).")
            for mdl, gm in g.groupby("model"):
                mh = sorted(map(str, gm["model_hash"].dropna().unique()))
                if len(mh) > 1:
                    raise SystemExit(f"{ds}/{mdl}: {len(mh)} distinct "
                                     f"model_hash across files ({mh}).")
            if seed_col and "partition_hash" in g.columns:
                ph_seeds = defaultdict(set)
                seed_parts = defaultdict(set)
                for _, row in g[[seed_col, "partition_hash"]].dropna().iterrows():
                    ph_seeds[str(row["partition_hash"])].add(int(row[seed_col]))
                    seed_parts[int(row[seed_col])].add(str(row["partition_hash"]))
                reused = {ph: sorted(sd) for ph, sd in ph_seeds.items()
                          if len(sd) > 1}
                if reused:
                    raise SystemExit(f"Dataset {ds}: partition_hash reused by "
                                     f"different seeds {list(reused.items())[:3]}.")
                mixed = [sd for sd, ph in seed_parts.items() if len(ph) > 1]
                if mixed:
                    raise SystemExit(f"Dataset {ds}: seed(s) {sorted(mixed)[:5]} "
                                     f"have DIFFERENT partition_hash across "
                                     f"models.")
    return frame


def main():
    ap = argparse.ArgumentParser(
        description="Canonical NB-corrected statistics (values from the CSVs).")
    src = ap.add_argument_group("input (default: auto-locate results/)")
    src.add_argument("--results-dir", default=None)
    src.add_argument("--csv", dest="csv_path", default=None)
    ap.add_argument("--metric", default="test_F1")
    ap.add_argument("--reference", default="M1")
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    ap.add_argument("--test-frac", type=float, default=TEST_FRAC)
    ap.add_argument("--equiv-margin", type=float, default=0.5)
    ap.add_argument("--extra-contrast", action="append", default=[],
                    metavar="A:B")
    ap.add_argument("--emit-latex", action="store_true")
    ap.add_argument("--save-latex", nargs="?", const=DEFAULT_LATEX_FILE,
                    default=None, metavar="FILE")
    ap.add_argument("--require-complete", action="store_true",
                    help="abort unless every model/dataset has all "
                         "30 seeds (42..71) -- for the final tables")
    ap.add_argument("--allow-legacy-results", action="store_true",
                    help="analyse CSVs that lack the provenance columns "
                         "(loses partition/protocol integrity guarantees)")
    args = ap.parse_args()

    if args.results_dir and args.csv_path:
        ap.error("provide at most one of --results-dir or --csv")
    extras = [tuple(c.split(":")) for c in args.extra_contrast]
    for c in extras:
        if len(c) != 2:
            ap.error(f"--extra-contrast must be A:B, got '{':'.join(c)}'")

    if args.csv_path:
        table, meta = load_long_csv(args.csv_path, args.metric,
                                    args.allow_legacy_results)
    else:
        table, meta = load_results_dir(_locate_results_dir(args.results_dir),
                                       args.metric, args.allow_legacy_results)
    if args.require_complete:
        require_complete_study(table)
    run_analysis(table, args.reference, extras, args.train_frac,
                 args.test_frac, args.equiv_margin, args.metric,
                 args.emit_latex, save_path=args.save_latex, meta=meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
