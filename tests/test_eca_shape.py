#!/usr/bin/env python3
"""
Verify that eca_block produces PER-CHANNEL attention. Requires TensorFlow;
skips cleanly (reported as SKIPPED by pytest, not passed) when it is absent.
The model_variants import is NOT wrapped in a broad except, so a genuine bug
in the ECA code surfaces as a failure rather than a false skip.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_eca_is_per_channel():
    # Skip (not pass) if TensorFlow is unavailable.
    pytest.importorskip("tensorflow",
                        reason="TensorFlow required for the ECA runtime test")
    import numpy as np
    from tensorflow.keras import layers, Model
    from model_variants import eca_block          # NOT inside a try/except

    C = 64
    inp = layers.Input(shape=(6, 6, C))
    model = Model(inp, eca_block(inp))
    assert model.output_shape[-1] == C, model.output_shape

    x = np.random.default_rng(0).random((2, 6, 6, C)).astype("float32")
    y = model.predict(x, verbose=0)
    ratio = (y + 1e-9) / (x + 1e-9)
    per_channel = ratio.mean(axis=(0, 1, 2))
    assert per_channel.std() > 1e-6, \
        "gate is constant across channels -> not per-channel attention"


if __name__ == "__main__":
    import importlib.util
    if importlib.util.find_spec("tensorflow") is None:
        print("[skip] TensorFlow not installed; run via pytest for a proper skip")
        sys.exit(0)
    test_eca_is_per_channel()
    print("[pass] eca_block applies distinct per-channel weights")
