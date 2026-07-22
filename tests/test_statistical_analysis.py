#!/usr/bin/env python3
"""Smoke tests for the canonical analysis. Run with pytest or plain python;
needs only numpy/pandas/scipy (no TensorFlow, no datasets)."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import statistical_analysis as sa   # noqa: E402


import shutil
_CACHE = None


def _make(tmp):
    """Generate the 16 synthetic CSVs ONCE (module cache), then copy into the
    per-test directory -- much faster than regenerating via subprocess."""
    global _CACHE
    if _CACHE is None:
        _CACHE = Path(tempfile.mkdtemp())
        subprocess.check_call([sys.executable,
                               str(REPO / "tests" / "generate_synthetic_results.py"),
                               "--out", str(_CACHE), "--n-runs", "30"])
    for f in _CACHE.glob("*.csv"):
        shutil.copy(f, tmp)


def test_load_and_analyse(tmp_path):
    _make(tmp_path)
    table, meta = sa.load_results_dir(tmp_path, metric="test_F1")
    assert len(table["HIKARI"]["M1"]) == 30
    assert meta["HIKARI"]["sizes"][("M1", 42)][0] > 100   # NB ratio from data (keyed by model,seed)
    tex = tmp_path / "out.tex"
    sa.run_analysis(table, "M1", [("M5", "M5b")], 0.70, 0.15, 0.5, "test_F1",
                    False, save_path=str(tex), meta=meta)
    assert tex.exists() and tex.stat().st_size > 0


def test_pairing_by_seed_intersection():
    table = {"DS": {"M1": {s: 0.9 for s in range(42, 72)},
                    "M2": {s: 0.5 for s in range(44, 69)}}}
    sa.run_analysis(table, "M1", [], 0.70, 0.15, 0.5, "test_F1", False)


def test_constant_difference_does_not_crash():
    # identical models (zero diff) and a constant non-zero diff.
    table = {"DS": {"M1": {s: 0.8 for s in range(42, 72)},
                    "M2": {s: 0.8 for s in range(42, 72)},
                    "M3": {s: 0.7 for s in range(42, 72)}}}
    sa.run_analysis(table, "M1", [], 0.70, 0.15, 0.5, "test_F1", False)


def test_duplicate_seed_aborts(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M1_HIKARI.csv"
    df = pd.read_csv(f)
    pd.concat([df, df.iloc[[0]]], ignore_index=True).to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_cross_model_protocol_mismatch_aborts(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M4_HIKARI.csv"       # give M4 a different protocol
    df = pd.read_csv(f)
    df["protocol_hash"] = "OTHERPROTO01"
    df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_long_csv_multi_dataset_ok(tmp_path):
    _make(tmp_path)
    merged = pd.concat([pd.read_csv(p) for p in tmp_path.glob("*.csv")],
                       ignore_index=True)
    m = tmp_path / "merged.csv"
    merged.to_csv(m, index=False)
    table, meta = sa.load_long_csv(str(m), metric="test_F1")   # must NOT reject
    assert "HIKARI" in table and "CIRA" in table




def test_missing_provenance_rejected(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M1_HIKARI.csv"
    df = pd.read_csv(f).drop(columns=["protocol_hash"])
    df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")           # strict
    sa.load_results_dir(tmp_path, metric="test_F1", allow_legacy=True)  # ok


def test_cross_file_duplicate_rejected(tmp_path):
    _make(tmp_path)
    import shutil
    shutil.copy(tmp_path / "M1_HIKARI.csv", tmp_path / "M1_HIKARI_backup.csv")
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_partition_mismatch_aborts(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M4_HIKARI.csv"
    df = pd.read_csv(f)
    df.loc[df.index[0], "partition_hash"] = "DIFFERENTPART"
    df.to_csv(f, index=False)
    # Caught at load time by the per-(dataset,seed) partition consistency check.
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


@pytest.mark.parametrize("col", ["protocol_hash", "partition_hash",
                                  "model_hash", "dataset_hash"])
def test_null_provenance_rejected(tmp_path, col):
    _make(tmp_path)
    f = tmp_path / "M1_HIKARI.csv"
    df = pd.read_csv(f); df.loc[df.index[0], col] = np.nan
    df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_mixed_model_hash_rejected(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M1_HIKARI.csv"
    df = pd.read_csv(f); df.loc[df.index[0], "model_hash"] = "OTHERMODEL01"
    df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_mixed_dataset_hash_rejected(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M0_HIKARI.csv"      # different dataset_hash than the rest
    df = pd.read_csv(f); df["dataset_hash"] = "OTHERDATA01"
    df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_nan_metric_rejected(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M1_CIRA.csv"
    df = pd.read_csv(f); df.loc[df.index[0], "test_F1"] = np.nan
    df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_metric_out_of_range_rejected(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M1_CIRA.csv"
    df = pd.read_csv(f); df.loc[df.index[0], "test_F1"] = 1.5
    df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_non_positive_partition_size_rejected(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M1_CIRA.csv"
    df = pd.read_csv(f); df.loc[df.index[0], "n_train"] = -10
    df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_partition_sizes_mismatch_across_models_aborts(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M4_HIKARI.csv"          # give M4 different sizes
    df = pd.read_csv(f); df["n_test"] = df["n_test"] + 1
    df.to_csv(f, index=False)
    table, meta = sa.load_results_dir(tmp_path, metric="test_F1")
    with pytest.raises(SystemExit):
        sa.run_analysis(table, "M1", [], 0.70, 0.15, 0.5, "test_F1", False,
                        meta=meta)


def test_partial_training_skips_statistics(tmp_path, monkeypatch):
    # Only M5b present -> run_statistics(tolerant=True) must NOT raise and
    # must return False (no reference M1 / not enough contrasts).
    import ablation_runner as ar
    _make(tmp_path)
    for pth in tmp_path.glob("*.csv"):
        if not pth.name.startswith("M5b_"):
            pth.unlink()
    monkeypatch.setattr(ar, "RESULTS_DIR", tmp_path)
    assert ar.run_statistics(tolerant=True) is False


def test_mixed_model_hash_across_files_rejected(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M1_HIKARI.csv"
    df = pd.read_csv(f)
    half = len(df) // 2
    df.loc[df.index[:half], "model_hash"] = "MODELHASH_A0"
    df.loc[df.index[half:], "model_hash"] = "MODELHASH_B0"
    # write as two files with disjoint seeds, no duplicate seeds
    df.iloc[:half].to_csv(tmp_path / "M1_HIKARI.csv", index=False)
    df.iloc[half:].to_csv(tmp_path / "M1b_HIKARI.csv", index=False)
    # rename model column of the 2nd so it's the SAME model split across files
    d2 = pd.read_csv(tmp_path / "M1b_HIKARI.csv"); d2["model"] = "M1"
    d2.to_csv(tmp_path / "M1b_HIKARI.csv", index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_repeated_partition_across_seeds_rejected(tmp_path):
    _make(tmp_path)
    for f in tmp_path.glob("*_HIKARI.csv"):
        df = pd.read_csv(f)
        df["partition_hash"] = "SAMEPART_HIKARI"       # every seed identical
        df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_fractional_seed_rejected(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M1_CIRA.csv"
    df = pd.read_csv(f); df["seed"] = df["seed"].astype(float)
    df.loc[df.index[0], "seed"] = 42.9
    df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_blank_hash_rejected(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M1_HIKARI.csv"
    df = pd.read_csv(f); df["model_hash"] = "   "
    df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_empty_csv_clear_error(tmp_path):
    _make(tmp_path)
    (tmp_path / "M9_HIKARI.csv").write_text("")     # zero bytes
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_stats_only_rejects_incomplete_files(tmp_path, monkeypatch):
    # A file missing tstar_F1 is INCOMPLETE per the canonical validator: the
    # strict analysis must abort (not silently succeed on test_F1 alone).
    import ablation_runner as ar
    _make(tmp_path)
    for f in tmp_path.glob("*.csv"):
        df = pd.read_csv(f)
        df.drop(columns=[c for c in df.columns if c.startswith("tstar_")],
                inplace=True)
        df.to_csv(f, index=False)
    monkeypatch.setattr(ar, "RESULTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        ar.run_statistics(tolerant=False)


def test_stats_only_requires_both_metrics(tmp_path, monkeypatch):
    # With complete files, strict stats runs both metrics and returns True.
    import ablation_runner as ar
    _make(tmp_path)
    monkeypatch.setattr(ar, "RESULTS_DIR", tmp_path)
    assert ar.run_statistics(tolerant=False) is True


def test_run_seed_mismatch_rejected(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M1_HIKARI.csv"
    df = pd.read_csv(f); df.loc[df.index[0], "run"] = 999
    df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_bad_tstar_rejected(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M1_CIRA.csv"
    df = pd.read_csv(f); df.loc[df.index[0], "t_star"] = 1.5
    df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_inconsistent_fractions_rejected(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M1_CIRA.csv"
    df = pd.read_csv(f); df.loc[df.index[0], "frac_train"] = 0.10   # != n_train/total
    df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_preflight_rejects_stray_csv(tmp_path, monkeypatch):
    import ablation_runner as ar
    from types import SimpleNamespace
    _make(tmp_path)
    (tmp_path / "M1_HIKARI_backup.csv").write_text(
        (tmp_path / "M1_HIKARI.csv").read_text())     # stray, non-canonical name
    monkeypatch.setattr(ar, "RESULTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        ar._preflight([], {}, SimpleNamespace())


def test_feature_order_hash_mismatch_aborts(tmp_path, monkeypatch):
    import ablation_runner as ar, json
    monkeypatch.setattr(ar, "RESULTS_DIR", tmp_path)
    (tmp_path / "feature_order_HIKARI.json").write_text(
        json.dumps({"dataset_hash": "OLDHASH"}))
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(SystemExit):
        ar._write_feature_order("HIKARI", X, "NEWHASH")


def test_dose1_requires_both_m5_m5b(tmp_path):
    import extended_analysis as ea
    _make(tmp_path)
    df = ea.load_runner_long(str(tmp_path))
    df_no_m5b = df[df["model"] != "M5b"]           # drop the M5b variant
    grad = ea.eca_gradient(df_no_m5b)
    for _, r in grad.iterrows():
        assert "1" not in r["doses_used"].split("+"), "dose 1 used M5 alone"


def test_latex_metric_label_dynamic(tmp_path):
    _make(tmp_path)
    table, meta = sa.load_results_dir(tmp_path, "test_AUC")
    tex = tmp_path / "auc.tex"
    sa.run_analysis(table, "M1", [], 0.70, 0.15, 0.5, "test_AUC", False,
                    save_path=str(tex), meta=meta)
    txt = tex.read_text()
    assert "AUC" in txt and "test-set F1" not in txt


def test_preflight_rejects_filename_content_mismatch(tmp_path, monkeypatch):
    import ablation_runner as ar
    from types import SimpleNamespace
    _make(tmp_path)
    df = pd.read_csv(tmp_path / "M2_HIKARI.csv")
    df["model"] = "M1"                     # file says M2, column says M1
    df.to_csv(tmp_path / "M2_HIKARI.csv", index=False)
    monkeypatch.setattr(ar, "RESULTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        ar._preflight([], {}, SimpleNamespace())


def test_malformed_feature_order_rejected(tmp_path, monkeypatch):
    import ablation_runner as ar
    monkeypatch.setattr(ar, "RESULTS_DIR", tmp_path)
    (tmp_path / "feature_order_HIKARI.json").write_text("{broken json")
    with pytest.raises(SystemExit):
        ar._write_feature_order("HIKARI", pd.DataFrame({"a": [1, 2]}), "H")


def test_auc_delta_header_is_dynamic(tmp_path):
    _make(tmp_path)
    table, meta = sa.load_results_dir(tmp_path, "test_AUC")
    tex = tmp_path / "auc.tex"
    sa.run_analysis(table, "M1", [], 0.70, 0.15, 0.5, "test_AUC", False,
                    save_path=str(tex), meta=meta)
    txt = tex.read_text()
    assert "\\Delta$AUC" in txt and "\\Delta$F1" not in txt


def test_transformer_params_in_fingerprint():
    # Regression: the Transformer architecture params must be in the runner's
    # HYPER (protocol hash) and config.py must be in the code-files hash, so a
    # change to HEAD_SIZE/NUM_HEADS/FF_DIM alters the fingerprints.
    import ablation_runner as ar
    import pathlib
    for k in ("head_size", "num_heads", "ff_dim"):
        assert k in ar.HYPER, f"{k} missing from HYPER"
    src = pathlib.Path(ar.__file__).read_text()
    assert '"config.py"' in src, "config.py not in _code_files_hash"


def test_require_complete_accepts_full(tmp_path):
    _make(tmp_path)                       # 30 seeds, all models, both datasets
    table, _ = sa.load_results_dir(tmp_path, "test_F1")
    sa.require_complete_study(table)      # must not raise


def test_require_complete_rejects_partial(tmp_path):
    subprocess.check_call([sys.executable,
                           str(REPO / "tests" / "generate_synthetic_results.py"),
                           "--out", str(tmp_path), "--n-runs", "5"])
    table, _ = sa.load_results_dir(tmp_path, "test_F1")
    with pytest.raises(SystemExit):
        sa.require_complete_study(table)


def test_preflight_accepts_empty_results_dir(tmp_path, monkeypatch):
    # A fresh campaign (no CSVs yet) MUST pass preflight -- regression guard
    # for the v7.6 bug where an empty dir aborted before any training.
    import ablation_runner as ar
    from types import SimpleNamespace
    monkeypatch.setattr(ar, "RESULTS_DIR", tmp_path)
    ar._preflight([], {}, SimpleNamespace())        # must NOT raise


def test_runner_stats_only_require_complete_partial(tmp_path, monkeypatch):
    import ablation_runner as ar
    subprocess.check_call([sys.executable,
                           str(REPO / "tests" / "generate_synthetic_results.py"),
                           "--out", str(tmp_path), "--n-runs", "5"])
    monkeypatch.setattr(ar, "RESULTS_DIR", tmp_path)
    # tolerant=False + require_complete: partial data must fail (return False
    # or raise), never silently produce "final" tables.
    with pytest.raises(SystemExit):
        ar.run_statistics(tolerant=False, require_complete=True)


def test_cross_dataset_study_drift_rejected(tmp_path):
    # HIKARI and CIRA produced under different study-wide configurations must
    # be refused, even though each dataset is internally consistent.
    _make(tmp_path)
    for f in tmp_path.glob("*_CIRA.csv"):
        df = pd.read_csv(f)
        df["study_protocol_hash"] = "OLD_CIRA_STUDY"
        df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


def test_global_code_version_mismatch_rejected(tmp_path):
    _make(tmp_path)
    f = tmp_path / "M4_HIKARI.csv"
    df = pd.read_csv(f)
    df["code_version"] = "bogus-version"      # internally consistent file
    df.to_csv(f, index=False)
    with pytest.raises(SystemExit):
        sa.load_results_dir(tmp_path, metric="test_F1")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
