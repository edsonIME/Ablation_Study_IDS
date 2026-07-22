#!/usr/bin/env python3
"""Regression suite for the group-aware split -- the component that carried
the critical v3 bug. Needs numpy/pandas/scikit-learn (no TensorFlow)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ablation_runner import split_indices, TRAIN_FRAC, VAL_FRAC, TEST_FRAC  # noqa: E402


def _toy(n_groups=90, seed=0):
    """Feature matrix with many groups of varied sizes and both classes well
    represented, so every partition can receive both classes."""
    rng = np.random.default_rng(seed)
    base = rng.random((n_groups, 6))
    labels = (np.arange(n_groups) % 2)          # balanced across groups
    counts = rng.integers(2, 20, n_groups)
    rows, ys = [], []
    for v, lab, c in zip(base, labels, counts):
        rows += [v] * c
        ys += [int(lab)] * c
    X = pd.DataFrame(rows, columns=[f"f{i}" for i in range(6)])
    return X, pd.Series(ys)


def _groups(X):
    return pd.util.hash_pandas_object(X, index=False).to_numpy()


def test_same_seed_is_reproducible():
    X, y = _toy()
    a = split_indices(X, y, 42, mode="group")
    b = split_indices(X, y, 42, mode="group")
    assert a[3]["partition_hash"] == b[3]["partition_hash"]
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2])


def test_different_seeds_change_partition():
    X, y = _toy()
    hashes = {split_indices(X, y, s, mode="group")[3]["partition_hash"]
              for s in range(42, 52)}
    assert len(hashes) >= 8, f"seeds barely change the split: {len(hashes)}/10"


def test_partition_hash_changes_with_seed():
    X, y = _toy()
    h42 = split_indices(X, y, 42, mode="group")[3]["partition_hash"]
    h43 = split_indices(X, y, 43, mode="group")[3]["partition_hash"]
    assert h42 != h43


def test_groups_never_cross_splits():
    X, y = _toy()
    g = _groups(X)
    for s in range(42, 47):
        mt, mv, me, _ = split_indices(X, y, s, mode="group")
        for gid in np.unique(g):
            where = [name for name, m in (("t", mt), ("v", mv), ("e", me))
                     if m[g == gid].any()]
            assert len(where) == 1, f"group {gid} in {where} at seed {s}"


def test_partitions_disjoint_and_exhaustive():
    X, y = _toy()
    for s in range(42, 47):
        mt, mv, me, _ = split_indices(X, y, s, mode="group")
        assert not (mt & mv).any() and not (mt & me).any() and not (mv & me).any()
        assert (mt | mv | me).all()


def test_each_partition_contains_both_classes():
    X, y = _toy()
    ya = y.to_numpy()
    for s in range(42, 47):
        mt, mv, me, _ = split_indices(X, y, s, mode="group")
        for m in (mt, mv, me):
            assert np.unique(ya[m]).size == 2, f"single-class partition at {s}"


def test_row_fractions_within_tolerance():
    X, y = _toy()
    for s in range(42, 47):
        mt, mv, me, info = split_indices(X, y, s, mode="group")
        for frac, target in ((info["frac"][0], TRAIN_FRAC),
                             (info["frac"][1], VAL_FRAC),
                             (info["frac"][2], TEST_FRAC)):
            assert abs(frac - target) <= 0.06, info["frac"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
