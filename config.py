"""
config.py
=========
Single source of truth for the training hyper-parameters, imported by BOTH
model_variants.py (which builds the models) and ablation_runner.py (which
records them in the protocol hash). Keeping them here prevents the two files
from drifting apart -- e.g. changing the learning rate in one place but not
the other. This module imports nothing heavy, so the runner can read it
without pulling in TensorFlow (needed for --dry-run / --stats-only).
"""

LEARNING_RATE = 0.001
L2_REG = 0.01
DROPOUT = 0.3

# Transformer encoder settings.
HEAD_SIZE = 128          # key_dim PER HEAD in Keras (num_heads * head_size total)
NUM_HEADS = 4
FF_DIM = 256

# Training-loop callbacks.
ES_PATIENCE = 10
RLROP_PATIENCE = 5
RLROP_FACTOR = 0.5
MIN_LR = 1e-6

# Resampling / split.
SMOTE_K = 5
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Monte Carlo schedule (single source; imported by runner and analysis).
BASE_SEED = 42
N_RUNS = 30
EXPECTED_SEEDS = tuple(range(BASE_SEED, BASE_SEED + N_RUNS))   # 42..71
