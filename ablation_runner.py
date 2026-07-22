#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 ABLATION RUNNER -- training driver for the CNN-ECA-Transformer ablation
===============================================================================
 Builds every variant from model_variants.py, trains each with Monte Carlo
 Cross-Validation (each seed produces a DIFFERENT group-aware partition), and
 delegates all statistics to statistical_analysis.py.

 Reproducibility / provenance
 ----------------------------
 Every per-run row records the resolved protocol: real partition sizes
 (n_train/n_val/n_test), a per-seed partition_hash, and protocol / model /
 dataset hashes that include the split strategy, hyper-parameters, the
 architecture version, the code version and the key library versions. On
 resume the runner refuses to append runs whose protocol differs (and refuses
 files with no protocol hash at all). The analysis additionally refuses to
 compare models trained under different protocols.

 Note on determinism: seeds fix the partition and the numpy/TF RNGs, and
 op-determinism is requested, but exact bit-for-bit reproducibility across
 GPUs/driver/TensorFlow builds is not guaranteed. Staging with --models
 preserves the same seed schedule and protocol, not identical floats.
===============================================================================
"""

import argparse
import hashlib
import importlib.metadata as _md
import json
import platform
import re
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (LEARNING_RATE, L2_REG, DROPOUT, HEAD_SIZE,
                    NUM_HEADS, FF_DIM, ES_PATIENCE, RLROP_PATIENCE,
                    RLROP_FACTOR, MIN_LR, SMOTE_K, TRAIN_FRAC,
                    VAL_FRAC, TEST_FRAC, BASE_SEED, N_RUNS)

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

CODE_VERSION = "v7.9"
RESULTS_DIR = Path("results")

SPLIT_TOLERANCE = 0.05
SPLIT_CANDIDATES = 8          # seeded candidate allocations; best-balanced wins

# Hyper-parameters that define the protocol (kept in sync with model_variants
# and single_run). They enter the protocol hash so a change forces a new file.
HYPER = {"lr": LEARNING_RATE, "l2": L2_REG, "dropout": DROPOUT,
         "head_size": HEAD_SIZE, "num_heads": NUM_HEADS, "ff_dim": FF_DIM,
         "es_patience": ES_PATIENCE, "rlrop_patience": RLROP_PATIENCE,
         "rlrop_factor": RLROP_FACTOR, "min_lr": MIN_LR, "smote_k": SMOTE_K,
         "train_frac": TRAIN_FRAC, "val_frac": VAL_FRAC, "test_frac": TEST_FRAC,
         "base_seed": BASE_SEED, "thr_grid": "0.01:0.99:0.01"}

KNOWN_DATASETS = ("HIKARI", "CIRA")
KNOWN_MODELS = ("M0", "M1", "M2", "M3", "M4", "M5", "M5b", "M6")

PRIORITY = [
    ("M4", "HIKARI"), ("M3", "HIKARI"), ("M4", "CIRA"),
    ("M5", "HIKARI"), ("M6", "HIKARI"),
    ("M5", "CIRA"), ("M6", "CIRA"),
    ("M5b", "HIKARI"), ("M5b", "CIRA"),
    ("M0", "HIKARI"), ("M1", "HIKARI"), ("M2", "HIKARI"),
    ("M0", "CIRA"), ("M1", "CIRA"), ("M2", "CIRA"), ("M3", "CIRA"),
]

DESC = {
    "M0":  "2 CNN + 2 ECA + Transformer (shallow baseline)",
    "M1":  "3 CNN + 2 ECA + Transformer (reference configuration)",
    "M2":  "4 CNN + 2 ECA + Transformer (depth +1)",
    "M3":  "3 CNN + 2 ECA, no Transformer (Flatten, dimensionally paired)",
    "M4":  "3 CNN + 0 ECA + Transformer (full ECA ablation)",
    "M5":  "3 CNN + 1 ECA (block 1) + Transformer (dose 1)",
    "M5b": "3 CNN + 1 ECA (block 2) + Transformer (dose 1, mirror of M5)",
    "M6":  "3 CNN + 3 ECA + Transformer (dose 3)",
}


def _sha(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True,
                                   default=str).encode()).hexdigest()[:12]


def _ver(pkg):
    try:
        return _md.version(pkg)
    except Exception:
        return "NA"


# =============================================================================
# 1. MODELS  (single source of truth: model_variants.py)
# =============================================================================
def _safe_read_csv(path):
    """Read a CSV, turning pandas' empty/parse errors into a clear message
    (mirrors statistical_analysis._safe_read_csv)."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise SystemExit(f"{path}: CSV file is empty or truncated.")
    except pd.errors.ParserError as exc:
        raise SystemExit(f"{path}: malformed CSV: {exc}")


def _make_builder(model_name):
    from model_variants import build_variant, VARIANTS
    if model_name not in VARIANTS:
        raise SystemExit(f"unknown model '{model_name}'; "
                         f"known: {', '.join(sorted(VARIANTS))}")
    spec = VARIANTS[model_name]
    return lambda shape: build_variant(shape, *spec)


def _model_hash(model_name):
    from model_variants import VARIANTS, ARCH_VERSION
    return _sha({"spec": VARIANTS[model_name], "arch": ARCH_VERSION,
                 "code": CODE_VERSION, "head_size": HEAD_SIZE,
                 "num_heads": NUM_HEADS, "ff_dim": FF_DIM, "dropout": DROPOUT,
                 "l2": L2_REG})


def _study_protocol_hash(split_mode, dedup, max_samples, epochs, batch_size):
    """Hash of the configuration COMMON to the whole study (no dataset
    specifics), so HIKARI and CIRA can be verified to share one general
    protocol: split strategy, hyper-parameters, epochs/batch, architecture
    version, code (files) version and library versions."""
    from model_variants import ARCH_VERSION
    payload = {**HYPER, "split": split_mode, "dedup": int(bool(dedup)),
               "max_samples": int(max_samples or 0), "epochs": int(epochs),
               "batch": int(batch_size), "arch": ARCH_VERSION,
               "code": CODE_VERSION, "code_files": _code_files_hash(),
               "python": platform.python_version(),
               "numpy": np.__version__, "pandas": pd.__version__,
               "scipy": _ver("scipy"), "tf": _ver("tensorflow"),
               "sklearn": _ver("scikit-learn"),
               "imblearn": _ver("imbalanced-learn")}
    return _sha(payload)


def _protocol_hash(split_mode, dedup, max_samples, epochs, batch_size,
                   n_features, dataset_hash):
    from model_variants import ARCH_VERSION
    payload = {**HYPER, "split": split_mode, "dedup": int(bool(dedup)),
               "max_samples": int(max_samples or 0), "epochs": int(epochs),
               "batch": int(batch_size), "n_features": int(n_features),
               "arch": ARCH_VERSION, "code": CODE_VERSION,
               "code_files": _code_files_hash(),
               "python": platform.python_version(),
               "numpy": np.__version__, "pandas": pd.__version__,
               "scipy": _ver("scipy"), "tf": _ver("tensorflow"),
               "sklearn": _ver("scikit-learn"),
               "imblearn": _ver("imbalanced-learn"), "data": dataset_hash}
    return _sha(payload)


def _code_files_hash():
    """Hash the source of the runner and the model definitions, so ANY code
    change alters the protocol hash even if the manual version strings are not
    bumped (a safety net, not a replacement for CODE_VERSION/ARCH_VERSION)."""
    h = hashlib.sha1()
    for mod in ("ablation_runner.py", "model_variants.py",
                "config.py"):
        f = Path(__file__).resolve().parent / mod
        try:
            h.update(f.read_bytes())
        except Exception:
            h.update(b"NA")
    return h.hexdigest()[:12]


# =============================================================================
# 2. DATA
# =============================================================================
# Exact identifier names (normalised: lowercased, non-alphanumerics stripped),
# covering common spellings: "Source IP", "Src IP", "src_ip" -> "sourceip"/
# "srcip"; "Destination Port" -> "destinationport"; etc.
_ID_EXACT = {
    "originh", "originp", "responh", "responp",
    "sourceip", "srcip", "destinationip", "dstip",
    "sourceport", "srcport", "destinationport", "dstport",
    "flowid",
}
# Substrings that unambiguously mark an identifier column.
_ID_SUBSTR = ("uid", "timestamp", "flowid", "unnamed")
LABEL_COLS = ("label", "traffic_category")


def _norm_col(col):
    return re.sub(r"[^a-z0-9]", "", str(col).lower())


def _is_identifier(col):
    n = _norm_col(col)
    return n in _ID_EXACT or any(tok in n for tok in _ID_SUBSTR)


def _clean_numeric(X):
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def dataset_fingerprint(X, y):
    """Stable short hash of the cleaned features AND labels (content, shape,
    column names, dtypes, class counts) -- not the features alone."""
    payload = {
        "shape": list(X.shape),
        "cols": list(map(str, X.columns)),
        "dtypes": [str(t) for t in X.dtypes],
        "Xhash": hashlib.sha1(
            pd.util.hash_pandas_object(X, index=False).to_numpy().tobytes()
        ).hexdigest(),
        "yhash": hashlib.sha1(
            pd.util.hash_pandas_object(pd.Series(np.asarray(y)),
                                       index=False).to_numpy().tobytes()
        ).hexdigest(),
        "n": int(len(y)), "pos": int(np.asarray(y).sum())}
    return f"{_sha(payload)}-{X.shape[0]}x{X.shape[1]}"


def load_dataset(name, path, dedup_features=False, max_samples=None):
    """Load HIKARI-2021 or CIRA-CIC-DoHBrw-2020 for BINARY classification.

    Order (identifiers removed BEFORE row dropping, so no row is discarded for
    a NaN in a column the model never uses): read -> (CIRA) keep
    Benign/Malicious -> drop identifiers by name -> coerce features to numeric
    -> drop rows with invalid label or features -> drop exact (feature+label)
    duplicates. Feature-space duplicates that remain are handled by the
    group-aware split; groups whose members carry conflicting labels are
    reported.
    """
    df = pd.read_csv(path, low_memory=False)
    n_raw = len(df)

    label_col = next((c for c in df.columns
                      if str(c).strip().lower() == "label"), None)
    if label_col is None:
        raise ValueError(f"{name}: 'Label' column not found in {path}")

    if not pd.api.types.is_numeric_dtype(df[label_col]):
        mapping = {"benign": 0, "malicious": 1}
        keep = df[label_col].astype(str).str.strip().str.lower().isin(mapping)
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"  [{name}] {n_drop:,} rows with non-binary label "
                  f"(DoH/NonDoH) discarded; task = Benign vs Malicious")
        df = df[keep].copy()
        df[label_col] = (df[label_col].astype(str).str.strip().str.lower()
                         .map(mapping))

    # Remove identifiers FIRST, then build features and labels.
    drop = [c for c in df.columns
            if _is_identifier(c) or str(c).strip().lower() in LABEL_COLS]
    y = pd.to_numeric(df[label_col], errors="coerce")
    X = _clean_numeric(df.drop(columns=drop))

    ok = X.notna().all(axis=1) & y.notna()
    X, y_num = X[ok], y[ok]
    # Validate the ORIGINAL numeric labels before casting: 0.5 -> 0 would
    # otherwise be truncated silently and accepted as "binary".
    invalid = ~y_num.isin([0, 1])
    if invalid.any():
        raise SystemExit(f"{name}: expected binary labels 0/1, found "
                         f"{sorted(y_num[invalid].unique().tolist())[:5]}.")
    X, y = X, y_num.astype("int32")

    # Drop exact (feature+label) duplicates -- truly redundant records.
    both = X.copy()
    both["__label__"] = y.to_numpy()
    keep_mask = ~both.duplicated()
    n_exact = int((~keep_mask).sum())
    if n_exact:
        print(f"  [{name}] {n_exact:,} exact (feature+label) duplicate rows "
              f"removed.")
    X, y = X[keep_mask].reset_index(drop=True), y[keep_mask].reset_index(drop=True)

    # Feature-space duplicate diagnostic + conflicting-label groups.
    dup_mask = X.duplicated()
    n_dup = int(dup_mask.sum())
    if n_dup:
        gid = pd.util.hash_pandas_object(X, index=False).to_numpy()
        conf_groups = pd.DataFrame({"g": gid, "y": y.to_numpy()}) \
            .groupby("g")["y"].nunique()
        n_conf_groups = int((conf_groups > 1).sum())
        print(f"  [{name}] note: {n_dup:,} rows ({100*n_dup/len(X):.1f}%) are "
              f"feature-space duplicates; kept together by the group-aware "
              f"split (no leakage). --dedup-features drops them instead.")
        if n_conf_groups:
            print(f"  [{name}] WARNING: {n_conf_groups:,} identical-feature "
                  f"groups carry CONFLICTING labels (irreducible ambiguity or "
                  f"a cleaning artefact); kept in one split, but review them.")
        if dedup_features:
            # Drop CONFLICTING-label groups entirely (never silently pick the
            # first label), then keep the first row of each consistent
            # duplicate group.
            conf_ids = set(conf_groups[conf_groups > 1].index)
            conf_row = np.array([g in conf_ids for g in gid])
            Xk, yk = X[~conf_row], y[~conf_row]
            dupf = Xk.duplicated()
            X = Xk[~dupf].reset_index(drop=True)
            y = yk[~dupf].reset_index(drop=True)
            print(f"  [{name}]        --dedup-features ACTIVE: dropped "
                  f"{int(conf_row.sum()):,} rows in conflicting-label groups "
                  f"and {int(dupf.sum()):,} consistent duplicate rows.")

    if max_samples and len(X) > max_samples:
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(X, y, train_size=int(max_samples),
                                      random_state=BASE_SEED, stratify=y)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        print(f"  [{name}] --max-samples: subsampled to {len(X):,} rows "
              f"(FUNCTIONAL smoke test only; not for reported results).")

    classes = set(int(v) for v in pd.unique(y))
    if classes != {0, 1}:
        raise SystemExit(f"{name}: only one class present ({sorted(classes)}); "
                         f"a binary task needs both.")

    pos = int(y.sum())
    print(f"  [{name}] {n_raw:,} raw rows -> {len(X):,} used | "
          f"{X.shape[1]} features")
    print(f"  [{name}] identifiers removed: {drop}")
    print(f"  [{name}] classes: 0 (benign) = {len(y)-pos:,} | 1 (attack) "
          f"= {pos:,} ({100*pos/len(y):.1f}% positives)")
    return X.reset_index(drop=True), y.reset_index(drop=True)


# =============================================================================
# 3. PARTITIONING (leakage-safe, row-balanced, and SEED-DEPENDENT)
# =============================================================================
def _greedy_assign(sizes, order, fracs):
    """Assign group indices (in `order`) to splits by largest ROW deficit."""
    total = float(sizes.sum())
    target = {k: fracs[k] * total for k in fracs}
    cur = {k: 0.0 for k in fracs}
    assign = {}
    for j in order:
        deficit = {k: target[k] - cur[k] for k in fracs}
        k = max(deficit, key=deficit.get)
        assign[int(j)] = k
        cur[k] += sizes[j]
    return assign


def split_indices(X, y, seed, mode="group",
                  train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """Return (train, val, test) boolean masks and an info dict.

    mode='group' (default, LEAKAGE-SAFE, ROW-BALANCED, SEED-DEPENDENT):
    identical feature vectors share a group; whole groups are assigned to one
    split only. For each seed, SPLIT_CANDIDATES seeded random group orders are
    tried and the allocation whose ROW fractions deviate least from
    train/val/test is kept. Different seeds explore different orders and
    therefore generally yield DIFFERENT partitions (this is what makes the 30
    runs genuine Monte Carlo resamples); the same seed always yields the same
    partition (reproducible). Allocation is stratified by group label, so both
    classes appear in every split when enough groups exist.

    mode='random' (DIAGNOSTIC ONLY, may leak): plain stratified row split.
    """
    from sklearn.model_selection import train_test_split
    n = len(X)
    idx = np.arange(n)
    test_frac = max(0.0, 1.0 - train_frac - val_frac)
    fracs = {"tr": train_frac, "val": val_frac, "te": test_frac}
    info = {"conflicts": 0}

    if mode == "random":
        tr, tmp = train_test_split(idx, test_size=1 - train_frac,
                                   random_state=seed, stratify=y)
        rel_test = test_frac / (1 - train_frac)
        val, te = train_test_split(tmp, test_size=rel_test,
                                   random_state=seed, stratify=y.iloc[tmp])
    elif mode == "group":
        groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
        gdf = pd.DataFrame({"g": groups, "y": np.asarray(y)})
        agg = gdf.groupby("g")["y"].agg(["mean", "size", "nunique"])
        info["conflicts"] = int((agg["nunique"] > 1).sum())
        agg["label"] = (agg["mean"] >= 0.5).astype(int)
        classes = {c: agg[agg["label"] == c] for c in (0, 1)}

        y_arr = np.asarray(y)
        global_pos = float(y_arr.mean())
        best_split_of, best_score = None, np.inf
        for attempt in range(SPLIT_CANDIDATES):
            rng = np.random.default_rng(seed * 1000 + attempt)
            assign_g = {}
            for c, sub in classes.items():
                if sub.empty:
                    continue
                gids = sub.index.to_numpy()
                sizes = sub["size"].to_numpy().astype(float)
                order = rng.permutation(len(gids))     # SEED-dependent order
                local = _greedy_assign(sizes, order, fracs)
                for j, split in local.items():
                    assign_g[gids[j]] = split
            split_of = np.array([assign_g[g] for g in groups])
            fr = {k: (split_of == k).mean() for k in fracs}
            size_dev = max(abs(fr[k] - fracs[k]) for k in fracs)
            # Reject candidates where any partition loses a class; otherwise
            # add the worst per-split deviation of the positive rate.
            invalid, class_dev = False, 0.0
            for k in fracs:
                yk = y_arr[split_of == k]
                if yk.size == 0 or np.unique(yk).size < 2:
                    invalid = True
                    break
                class_dev = max(class_dev, abs(yk.mean() - global_pos))
            score = (1e9 if invalid else 0.0) + size_dev + class_dev
            if score < best_score:
                best_score, best_split_of = score, split_of
        split_of = best_split_of
        tr = idx[split_of == "tr"]
        val = idx[split_of == "val"]
        te = idx[split_of == "te"]
    else:
        raise SystemExit(f"unknown split mode '{mode}' (use group|random)")

    m_tr = np.zeros(n, bool); m_tr[tr] = True
    m_val = np.zeros(n, bool); m_val[val] = True
    m_te = np.zeros(n, bool); m_te[te] = True
    info["frac"] = (m_tr.mean(), m_val.mean(), m_te.mean())
    payload = np.concatenate([np.where(m_tr)[0], [-1], np.where(m_val)[0],
                              [-1], np.where(m_te)[0]]).astype(np.int64)
    info["partition_hash"] = hashlib.sha1(payload.tobytes()).hexdigest()[:12]
    return m_tr, m_val, m_te, info


def _validate_partition(y_tr, y_val, y_te, seed):
    """Fail fast with a clear message if a partition is unusable."""
    for part, yy in (("train", y_tr), ("val", y_val), ("test", y_te)):
        if len(yy) == 0:
            raise SystemExit(f"seed {seed}: {part} partition is empty; "
                             f"the split cannot proceed (too few groups?).")
        if len(np.unique(yy)) < 2:
            raise SystemExit(f"seed {seed}: {part} partition has a single "
                             f"class; metrics like AUC are undefined. Check "
                             f"the group structure / class balance.")
    minority = int(np.bincount(np.asarray(y_tr)).min())
    if minority < SMOTE_K + 1:
        raise SystemExit(f"seed {seed}: only {minority} minority-class "
                         f"training samples; SMOTE(k={SMOTE_K}) needs "
                         f">= {SMOTE_K + 1}.")


# =============================================================================
# 4. METRICS
# =============================================================================
def metrics_at(y_true, proba, thr):
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)
    pred = (proba > thr).astype("int32")
    return {"Acc":  accuracy_score(y_true, pred),
            "Prec": precision_score(y_true, pred, zero_division=0),
            "Rec":  recall_score(y_true, pred, zero_division=0),
            "F1":   f1_score(y_true, pred, zero_division=0),
            "AUC":  roc_auc_score(y_true, proba)}


def best_threshold(y_val, proba_val):
    from sklearn.metrics import f1_score
    grid = np.arange(0.01, 1.00, 0.01)
    sc = [f1_score(y_val, (proba_val > t).astype("int32"), zero_division=0)
          for t in grid]
    return float(grid[int(np.argmax(sc))])


# =============================================================================
# 5. ONE MONTE CARLO RUN
# =============================================================================
def single_run(build_fn, X, y, seed, epochs, batch_size, split_mode):
    """Train one model on one seeded partition; return a config-stamped record.

    Pipeline: seeded group split -> validate partition -> fit MinMax on the
    RAW training rows -> transform train/val/test -> SMOTE on the SCALED
    training rows (so SMOTE neighbours use normalised distances) -> train.
    """
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from imblearn.over_sampling import SMOTE
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    tf.keras.backend.clear_session()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    try:                                   # best-effort op determinism
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    m_tr, m_val, m_te, info = split_indices(X, y, seed, mode=split_mode)
    X_tr, y_tr = X[m_tr], y[m_tr]
    X_val, y_val = X[m_val], y[m_val]
    X_te, y_te = X[m_te], y[m_te]
    n_train_raw, n_val, n_test = len(y_tr), len(y_val), len(y_te)
    _validate_partition(y_tr, y_val, y_te, seed)

    sc = MinMaxScaler().fit(X_tr)                     # fit on RAW train only
    X_tr_s = sc.transform(X_tr)
    X_val_s, X_te_s = sc.transform(X_val), sc.transform(X_te)

    X_tr_s, y_tr = SMOTE(k_neighbors=SMOTE_K, random_state=seed) \
        .fit_resample(X_tr_s, y_tr)                   # on SCALED train
    n_train_smote = len(y_tr)

    size = int(np.ceil(np.sqrt(X_tr_s.shape[1])))

    def to2d(a):
        return np.pad(a, ((0, 0), (0, size**2 - a.shape[1])), "constant") \
                 .reshape(-1, size, size, 1)

    Xtr2, Xva2, Xte2 = to2d(X_tr_s), to2d(X_val_s), to2d(X_te_s)

    model = build_fn((size, size, 1))
    cbs = [EarlyStopping(monitor="val_loss", patience=HYPER["es_patience"],
                         restore_best_weights=True),
           ReduceLROnPlateau(monitor="val_loss",
                             factor=HYPER["rlrop_factor"],
                             patience=HYPER["rlrop_patience"],
                             min_lr=HYPER["min_lr"])]

    t0 = time.perf_counter()
    hist = model.fit(Xtr2, y_tr, validation_data=(Xva2, y_val), epochs=epochs,
                     batch_size=batch_size, callbacks=cbs, verbose=0)
    train_time = time.perf_counter() - t0

    p_val = model.predict(Xva2, verbose=0).ravel()
    t0 = time.perf_counter()
    p_te = model.predict(Xte2, verbose=0).ravel()
    test_time = time.perf_counter() - t0

    t_star = best_threshold(y_val, p_val)
    rec = {"seed": seed,
           "n_params": int(model.count_params()),
           "grid": size,
           "tokens": (size // 2 // 2) ** 2,
           "best_epoch": int(np.argmin(hist.history["val_loss"]) + 1),
           "t_star": t_star,
           "train_time_s": train_time,
           "test_time_s": test_time,
           "n_train": n_train_raw, "n_val": n_val, "n_test": n_test,
           "n_train_smote": n_train_smote, "n_features": int(X.shape[1]),
           "frac_train": round(info["frac"][0], 4),
           "frac_val": round(info["frac"][1], 4),
           "frac_test": round(info["frac"][2], 4),
           "group_conflicts": int(info["conflicts"]),
           "partition_hash": info["partition_hash"],
           "split_mode": split_mode, "epochs_max": int(epochs),
           "batch_size": int(batch_size)}
    for k, v in metrics_at(y_te, p_te, 0.5).items():
        rec[f"test_{k}"] = v
    for k, v in metrics_at(y_te, p_te, t_star).items():
        rec[f"tstar_{k}"] = v
    return rec


# =============================================================================
# 6. TRAINING (resumable; refuses to mix protocols)
# =============================================================================
def _validate_resume_file(prev, csv, model_name, dataset, proto, mhash,
                          dataset_hash, study=None):
    """Fully validate an existing results file BEFORE resuming, applying the
    same strictness as the analyser (no dropna() leniency): every required
    column present and non-null, integer non-duplicate seeds, and EXACT
    single-valued matches of protocol/model/dataset hashes, code version,
    model and dataset. Returns the set of completed seeds."""
    # Full content/integrity validation via the SINGLE canonical definition
    # of a valid runner results file (schema completeness, non-null, integer
    # seeds/run, positive sizes, in-range metrics, per-file hash consistency,
    # no partition reuse across seeds, no duplicate rows).
    from statistical_analysis import validate_runner_results_file
    validate_runner_results_file(prev, str(csv))
    if "code_version" not in prev.columns:
        raise SystemExit(f"Refusing to append to {csv}: no code_version.")
    if prev["seed"].astype(int).duplicated().any():
        raise SystemExit(f"Refusing to append to {csv}: duplicate seed(s).")
    exp_run = prev["seed"].astype(int) - BASE_SEED + 1
    if not (prev["run"].astype(int) == exp_run).all():
        raise SystemExit(f"Refusing to append to {csv}: 'run' must equal "
                         f"seed - {BASE_SEED} + 1.")
    # Exact-match this configuration against the stored one.
    exact = {"protocol_hash": proto, "model_hash": mhash,
             "dataset_hash": dataset_hash, "code_version": CODE_VERSION,
             "model": model_name, "dataset": dataset}
    if study is not None and "study_protocol_hash" in prev.columns:
        exact["study_protocol_hash"] = study
    for col, want in exact.items():
        got = set(prev[col].astype(str))
        if got != {str(want)}:
            raise SystemExit(
                f"Refusing to append to {csv}: {col} = {sorted(got)} != "
                f"current '{want}'. These runs are not from the same "
                f"configuration; move or rename the file.")
    return set(prev["seed"].astype(int))


def train_pair(model_name, dataset, X, y, n_runs, epochs, batch_size,
               split_mode, dedup, max_samples, dataset_hash):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv = RESULTS_DIR / f"{model_name}_{dataset}.csv"

    proto = _protocol_hash(split_mode, dedup, max_samples, epochs, batch_size,
                           X.shape[1], dataset_hash)
    study = _study_protocol_hash(split_mode, dedup, max_samples, epochs,
                                 batch_size)
    mhash = _model_hash(model_name)

    done = set()
    if csv.exists():
        prev = _safe_read_csv(csv)
        done = _validate_resume_file(prev, csv, model_name, dataset, proto,
                                     mhash, dataset_hash, study)

    pending = [BASE_SEED + r for r in range(n_runs) if BASE_SEED + r not in done]

    print(f"\n--- {model_name} / {dataset} -- {DESC[model_name]}  "
          f"[proto {proto} | model {mhash}]")
    if not pending:
        print(f"    already complete ({len(done)}/{n_runs}); skipping")
    build_fn = _make_builder(model_name)
    for i, seed in enumerate(pending, 1):
        t0 = time.perf_counter()
        rec = single_run(build_fn, X, y, seed, epochs, batch_size, split_mode)
        rec.update({"model": model_name, "dataset": dataset,
                    "run": seed - BASE_SEED + 1,
                    "dedup_features": int(bool(dedup)),
                    "max_samples": int(max_samples or 0),
                    "dataset_hash": dataset_hash, "protocol_hash": proto,
                    "study_protocol_hash": study,
                    "model_hash": mhash, "code_version": CODE_VERSION})
        for frac, target, part in ((rec["frac_train"], TRAIN_FRAC, "train"),
                                   (rec["frac_val"], VAL_FRAC, "val"),
                                   (rec["frac_test"], TEST_FRAC, "test")):
            if abs(frac - target) > SPLIT_TOLERANCE:
                print(f"    [warn] seed {seed}: {part} fraction {frac:.3f} "
                      f"deviates > {SPLIT_TOLERANCE} from {target:.2f}; "
                      f"recorded and used as-is.")
        # Atomic append: build the full frame and replace the file in one
        # os.replace, so an interruption can never leave a half-written row.
        new_row = pd.DataFrame([rec])
        full = (pd.concat([pd.read_csv(csv), new_row], ignore_index=True)
                if csv.exists() else new_row)
        tmp = csv.with_name(csv.name + ".tmp")
        full.to_csv(tmp, index=False)
        os.replace(tmp, csv)
        print(f"    [{len(done)+i:02d}/{n_runs}] seed {seed} | "
              f"F1(0.5)={rec['test_F1']:.4f} | F1(t*)={rec['tstar_F1']:.4f} "
              f"(t*={rec['t_star']:.2f}) | tr/val/te="
              f"{rec['n_train']}/{rec['n_val']}/{rec['n_test']} | "
              f"part {rec['partition_hash']} | {time.perf_counter()-t0:.0f}s",
              flush=True)

    df = pd.read_csv(csv)
    print(f"    >> F1(0.5) = {df['test_F1'].mean():.4f} +/- {df['test_F1'].std():.4f}"
          f" | F1(t*) = {df['tstar_F1'].mean():.4f} +/- {df['tstar_F1'].std():.4f}"
          f" | {int(df['n_params'].iloc[0]):,} params")


# =============================================================================
# 7. STATISTICS  (delegated to the canonical module)
# =============================================================================
def run_statistics(tolerant=False, require_complete=False):
    """Run the canonical analysis for BOTH metrics.

    Integrity violations (bad/blank/inconsistent provenance, duplicates,
    out-of-range metrics, reused partitions) raise a plain SystemExit and are
    NEVER swallowed -- they abort here in both modes. Only IncompleteResults
    (reference model or >=5 shared seeds not present yet) is tolerated when
    tolerant=True (end of a partial training stage).

    Returns: strict mode -> True only if BOTH metrics were analysed;
    tolerant mode -> True if at least one metric was analysed."""
    from statistical_analysis import (load_results_dir, run_analysis,
                                      IncompleteResults, require_complete_study)
    outcomes = []
    for metric, tex in (("test_F1", "statistical_tables.tex"),
                        ("tstar_F1", "statistical_tables_tstar.tex")):
        try:
            table, meta = load_results_dir(RESULTS_DIR, metric)   # integrity -> raises
            if require_complete:
                require_complete_study(table)      # aborts if not all 30 seeds
            run_analysis(table, reference="M1", extra_contrasts=[("M5", "M5b")],
                         train_frac=TRAIN_FRAC, test_frac=TEST_FRAC,
                         equiv_margin_pp=0.5, metric=metric,
                         emit_latex=False, save_path=tex, meta=meta)
            outcomes.append(True)
        except IncompleteResults as err:
            print(f"[stats] {metric}: {err}")
            outcomes.append(False)
    if tolerant:
        if not any(outcomes):
            print("[stats] statistics skipped: not enough completed runs yet "
                  "(need M1 + another model with >= 5 shared seeds). The "
                  "per-run CSVs are saved; re-run --stats-only after more "
                  "stages.")
        return any(outcomes)
    return len(outcomes) == 2 and all(outcomes)     # strict: both required


# =============================================================================
# 8. MAIN
# =============================================================================
def _write_feature_order(ds, X, dhash):
    """Write results/feature_order_<ds>.json atomically. If one already exists
    with a DIFFERENT dataset_hash, abort rather than silently overwrite (it
    belongs to different data)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fp = RESULTS_DIR / f"feature_order_{ds}.json"
    if fp.exists():
        try:
            old = json.loads(fp.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{fp}: malformed feature-order JSON ({exc}); "
                             f"refusing to overwrite provenance. Remove it.")
        for k in ("dataset", "n_features", "features", "dataset_hash"):
            if k not in old:
                raise SystemExit(f"{fp}: missing key '{k}'; corrupt provenance.")
        if old["dataset_hash"] != dhash:
            raise SystemExit(f"{fp}: existing dataset_hash {old['dataset_hash']} "
                             f"!= current {dhash}. Refusing to overwrite.")
    payload = {"dataset": ds, "n_features": int(X.shape[1]),
               "features": list(map(str, X.columns)), "dataset_hash": dhash}
    tmp = fp.with_name(fp.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, fp)


def _preflight(plan, cache, args):
    """Validate the WHOLE results directory before any training. For EVERY
    *.csv (not only the models of the current stage): the name must be a
    canonical <MODEL>_<DATASET>.csv; the file must be structurally valid; its
    'model'/'dataset' columns must MATCH the file name; no (dataset,model,seed)
    may be duplicated across files; and -- for every dataset already loaded --
    the file must match the CURRENT protocol/model/dataset hashes. This catches
    an incompatible file of a not-yet-scheduled model in seconds, instead of
    only when the final analysis reads the whole directory."""
    from statistical_analysis import validate_runner_results_file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    print("\nPreflight: checking the results directory ...")
    if not csv_files:
        print("  no existing CSVs; fresh campaign.\nPreflight OK.\n")
        return
    valid = {f"{m}_{d}" for m in KNOWN_MODELS for d in KNOWN_DATASETS}
    seen = {}
    for csvp in csv_files:
        if csvp.stem not in valid:
            raise SystemExit(
                f"Preflight: unexpected file '{csvp.name}' in {RESULTS_DIR} "
                f"(expected <MODEL>_<DATASET>.csv). Remove stray/backup/merged "
                f"files before training.")
        exp_model, exp_dataset = csvp.stem.split("_", 1)
        df = _safe_read_csv(csvp)
        validate_runner_results_file(df, str(csvp))
        got_m = sorted(set(df["model"].astype(str)))
        got_d = sorted(set(df["dataset"].astype(str)))
        if got_m != [exp_model]:
            raise SystemExit(f"Preflight: {csvp.name} 'model' column {got_m} "
                             f"does not match the file name model "
                             f"'{exp_model}'.")
        if got_d != [exp_dataset]:
            raise SystemExit(f"Preflight: {csvp.name} 'dataset' column {got_d} "
                             f"does not match the file name dataset "
                             f"'{exp_dataset}'.")
        for sd in df["seed"].astype(int):
            key = (exp_dataset, exp_model, int(sd))
            if key in seen:
                raise SystemExit(f"Preflight: duplicate {key} in {csvp.name} "
                                 f"and {seen[key]}.")
            seen[key] = csvp.name
        if exp_dataset in cache:                 # validate protocol of ALL files
            X, y, dhash = cache[exp_dataset]      # of a loaded dataset, not just
            proto = _protocol_hash(args.split, args.dedup_features,  # planned ones
                                   args.max_samples, args.epochs,
                                   args.batch_size, X.shape[1], dhash)
            study = _study_protocol_hash(args.split, args.dedup_features,
                                         args.max_samples, args.epochs,
                                         args.batch_size)
            _validate_resume_file(df, csvp, exp_model, exp_dataset, proto,
                                  _model_hash(exp_model), dhash, study)
            print(f"  [ok] {csvp.name} compatible.")
    # Global cross-model integrity for EVERY dataset present (even ones not
    # loaded this run): the canonical loader enforces one protocol_hash /
    # dataset_hash / model_hash per dataset, identical partitions per seed, no
    # reused partitions and no cross-file duplicates.
    from statistical_analysis import load_results_dir
    load_results_dir(RESULTS_DIR, "test_F1")
    print("Preflight OK.\n")


def main():
    ap = argparse.ArgumentParser(
        description="Ablation training runner. Statistics are delegated to "
                    "statistical_analysis.py.")
    ap.add_argument("--results-dir", default="results",
                    help="directory for per-run CSVs (use a SEPARATE dir for "
                         "smoke tests so they never mix with the full campaign)")
    ap.add_argument("--hikari-csv", default="ALLFLOWMETER_HIKARI2021.csv")
    ap.add_argument("--cira-csv", default="CIRA.csv")
    ap.add_argument("--models", nargs="+", default=None,
                    help=f"subset of {', '.join(KNOWN_MODELS)}")
    ap.add_argument("--datasets", nargs="+", default=list(KNOWN_DATASETS))
    ap.add_argument("--n-runs", type=int, default=N_RUNS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--split", choices=("group", "random"), default="group")
    ap.add_argument("--dedup-features", action="store_true")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap rows after cleaning (FUNCTIONAL smoke test only)")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--require-complete", action="store_true",
                    help="with --stats-only: abort unless every "
                         "model/dataset has all 30 seeds (42..71)")
    ap.add_argument("--no-stats", action="store_true",
                    help="train only; skip the statistics at the end "
                         "(useful for smoke tests and staged runs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the execution plan and exit (trains nothing)")
    args = ap.parse_args()

    global RESULTS_DIR
    RESULTS_DIR = Path(args.results_dir)

    for name_, val in (("--n-runs", args.n_runs), ("--epochs", args.epochs),
                       ("--batch-size", args.batch_size)):
        if val < 1:
            raise SystemExit(f"{name_} must be at least 1.")
    if args.max_samples is not None and args.max_samples < 10:
        raise SystemExit("--max-samples is too small for a stratified "
                         "three-way split (use >= 10).")

    if args.models:
        unknown = [m for m in args.models if m not in KNOWN_MODELS]
        if unknown:
            raise SystemExit(f"Unknown model(s): {', '.join(unknown)}. "
                             f"Available: {', '.join(KNOWN_MODELS)}.")
    unknown_ds = [d for d in args.datasets if d not in KNOWN_DATASETS]
    if unknown_ds:
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown_ds)}. "
                         f"Available: {', '.join(KNOWN_DATASETS)}.")

    if args.stats_only:
        return 0 if run_statistics(
            require_complete=args.require_complete) else 1

    plan = [(m, d) for m, d in PRIORITY
            if d in args.datasets
            and (args.models is None or m in args.models)]

    print("=" * 78)
    print("EXECUTION PLAN (priority order)")
    print("=" * 78)
    for i, (m, d) in enumerate(plan, 1):
        print(f" {i:2}. {m:6} / {d:6} -- {DESC[m]}")
    print(f"split = {args.split}"
          + ("  (LEAKAGE-SAFE, row-balanced, seed-dependent)"
             if args.split == "group"
             else "  (WARNING: leaky diagnostic; not for results)"))
    if args.dry_run:
        return 0
    if not plan:
        print("Nothing to do.")
        return 0

    cache = {}
    for ds in {d for _, d in plan}:
        path = args.hikari_csv if ds == "HIKARI" else args.cira_csv
        if not path or not Path(path).exists():
            raise SystemExit(f"{ds} CSV not found: {path!r}. "
                             f"Use --hikari-csv / --cira-csv.")
        X, y = load_dataset(ds, path, dedup_features=args.dedup_features,
                            max_samples=args.max_samples)
        dhash = dataset_fingerprint(X, y)
        s = int(np.ceil(np.sqrt(X.shape[1])))
        if s < 4:
            raise SystemExit(f"{ds}: grid {s}x{s} collapses to 0 tokens after "
                             f"two poolings; need >= 16 features for this "
                             f"architecture.")
        tok = (s // 2 // 2) ** 2
        print(f"\n{'#'*78}")
        print(f"# {ds}: {X.shape[0]:,} samples | {X.shape[1]} features "
              f"| grid {s}x{s} | {tok} token(s) | data {dhash}")
        if tok == 1:
            print("# WARNING: 1 token -> self-attention is DEGENERATE here; the")
            print("# 'Transformer' block reduces to dense + LayerNorm + residual.")
            print("# Interpret M1-vs-M3 on this dataset as the contribution of")
            print("# the whole Transformer block, NOT of multi-token attention.")
        print("#" * 78)
        cache[ds] = (X, y, dhash)

    # Validate the whole results directory FIRST (strays, duplicates, protocol).
    _preflight(plan, cache, args)
    # Only now record the feature order (after preflight, with a hash guard).
    for ds, (X, y, dhash) in cache.items():
        _write_feature_order(ds, X, dhash)

    t_start = time.perf_counter()
    for m, d in plan:
        X, y, dhash = cache[d]
        train_pair(m, d, X, y, args.n_runs, args.epochs, args.batch_size,
                   args.split, args.dedup_features, args.max_samples, dhash)
    print(f"\nTraining finished in {(time.perf_counter()-t_start)/3600:.1f} h")

    if not args.no_stats:
        run_statistics(tolerant=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
