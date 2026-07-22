"""
model_variants.py
=================
Single source of truth for every ablation architecture. The training runner
imports ``eca_block``, ``transformer_encoder``, ``build_variant`` and
``VARIANTS`` from here, so the models trained and the models described in the
analysis can never diverge.

Design axes captured by the factory:
  * convolutional depth       (number of conv blocks);
  * channel attention (ECA)   (how many ECA modules, and where);
  * global attention          (Transformer encoder present or absent).

M5b derives from M1 by removing the FIRST ECA module and keeping the SECOND
(after the 64-filter block):

    M1  (2 ECA): Conv32-Pool-ECA -> Conv64-Pool-ECA -> Conv128 -> Transformer
    M5  (1 ECA): Conv32-Pool-ECA -> Conv64-Pool     -> Conv128 -> Transformer
    M5b (1 ECA): Conv32-Pool     -> Conv64-Pool-ECA -> Conv128 -> Transformer

Purpose (see Threats to Validity): M5 changes ECA COUNT and POSITION jointly.
M5b holds the count at 1 and mirrors the position, so:
    M5  vs M5b -> isolates the PLACEMENT effect at fixed count;
    M1  vs M5b -> marginal contribution of the 1st ECA module;
    M1  vs M5  -> marginal contribution of the 2nd ECA module.

Usage:
  A) Import into the runner:  from model_variants import VARIANTS, build_variant
  B) CLI: `python model_variants.py --size 9` prints parameter counts for all
     variants (input for the model-complexity table).
"""

# Bump this whenever the architecture code changes (e.g. the ECA block),
# so downstream protocol hashes differ and old/new runs are never mixed.
ARCH_VERSION = "eca-conv1d-v1"

import numpy as np
from tensorflow.keras import layers, Model
from config import (LEARNING_RATE, L2_REG, DROPOUT, HEAD_SIZE,
                    NUM_HEADS, FF_DIM)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.regularizers import l2


# ======================================================================
# Shared building blocks
# ======================================================================

def eca_block(input_tensor):
    """Efficient Channel Attention (Wang et al., CVPR 2020).

    Produces ONE attention weight PER CHANNEL via a 1-D convolution over the
    channel-descriptor sequence, then rescales each channel by its own
    weight. This is genuine per-channel attention -- not a single global
    gate.

    Pipeline:
        GlobalAveragePooling2D : (B, H, W, C) -> (B, C)   one descriptor/channel
        Reshape                : (B, C)       -> (B, C, 1) channels as a sequence
        Conv1D(filters=1, k)   : (B, C, 1)    -> (B, C, 1) local cross-channel mix
        sigmoid                : per-channel gate in (0, 1)
        Reshape                : (B, C, 1)    -> (B, 1, 1, C) broadcastable
        Multiply               : recalibrate each channel independently

    The adaptive kernel size follows the ECA-Net rule and is forced ODD (the
    convolution is centred on each channel), with a floor of 3.
    """
    channels = int(input_tensor.shape[-1])

    # Adaptive, ODD kernel size (ECA-Net gamma=2, b=1 heuristic).
    t = int(abs((np.log2(channels) + 1) / 2))
    kernel_size = t if t % 2 == 1 else t + 1
    kernel_size = max(3, kernel_size)

    # Squeeze: one global descriptor per channel.
    attention = layers.GlobalAveragePooling2D()(input_tensor)      # (B, C)

    # Treat the channel axis as a 1-D sequence of length C.
    attention = layers.Reshape((channels, 1))(attention)           # (B, C, 1)

    # Excitation: local cross-channel interaction, no dimensionality
    # reduction (the defining property of ECA vs SE attention). Output keeps
    # length C, i.e. one weight per channel.
    attention = layers.Conv1D(
        filters=1,
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,
    )(attention)                                                   # (B, C, 1)
    attention = layers.Activation("sigmoid")(attention)

    # Restore a broadcastable per-channel attention tensor and recalibrate.
    attention = layers.Reshape((1, 1, channels))(attention)        # (B,1,1,C)
    return layers.Multiply()([input_tensor, attention])


def transformer_encoder(inputs, head_size=HEAD_SIZE, num_heads=NUM_HEADS,
                        ff_dim=FF_DIM, dropout=DROPOUT):
    """Pre-norm Transformer encoder block.

    NOTE (report this exactly in the manuscript): ``key_dim=head_size`` is the
    dimensionality PER HEAD in Keras, so the internal projection is
    num_heads * head_size = 4 * 128 = 512 dims (NOT 128 split across 4 heads).
    NOTE: there is NO positional encoding; self-attention mixes tokens by
    content only. The later Flatten preserves an order for the dense head, but
    the attention itself receives no absolute grid position. Do not describe
    this block as positional spatial attention without stating this limitation.

    LayerNorm -> MultiHeadAttention -> Dropout -> residual, then
    LayerNorm -> Dense(ff) -> Dropout -> Dense(channels) -> residual.
    """
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    return x + res


def compile_reference(model):
    """Compile with the standardized settings shared by all variants."""
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=BinaryCrossentropy(),
        metrics=['accuracy', 'Precision', 'Recall', 'AUC']
    )
    return model


# ======================================================================
# Parametric factory: one specification -> any ablation variant
# ======================================================================

def build_variant(input_shape, conv_filters, eca_after, use_transformer):
    """Build any ablation variant from a declarative specification.

    Args:
        input_shape:     e.g. (9, 9, 1) for HIKARI-2021 (79 features
                         zero-padded to 81), (6, 6, 1) for CIRA-DoHBrw.
        conv_filters:    tuple of filters per block, e.g. (32, 64, 128).
                         Max-pooling is applied after blocks 1 and 2 only.
        eca_after:       set of 1-based block indices followed by an ECA
                         module, e.g. {1, 2} for M1, {2} for M5b.
        use_transformer: True  -> Reshape->Transformer->Flatten;
                         False -> Flatten head, dimensionally paired.

    Returns:
        A compiled Keras model with the standardized settings.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for i, filters in enumerate(conv_filters, start=1):
        x = layers.Conv2D(filters, (3, 3), padding='same',
                          activation='relu')(x)
        if i <= 2:                      # pooling on blocks 1-2 only
            x = layers.MaxPooling2D((2, 2))(x)
        if i in eca_after:              # channel attention placement
            x = eca_block(x)

    if use_transformer:
        x = layers.Reshape((-1, x.shape[-1]))(x)
        x = transformer_encoder(x)
        x = layers.Flatten()(x)
    else:
        # No Transformer: Flatten keeps the head dimensionally paired.
        x = layers.Flatten()(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(L2_REG))(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=l2(L2_REG))(x)
    output = layers.Dense(1, activation='sigmoid', name='binary_output')(x)

    return compile_reference(Model(inputs=inputs, outputs=output))


def build_mtl_model_M5b(input_shape, num_classes=1):
    """Convenience wrapper for M5b via the factory (kept as a convenience wrapper)."""
    return build_variant(input_shape, *VARIANTS["M5b"])


# Variant registry: model name -> (conv_filters, eca_after, use_transformer).
VARIANTS = {
    # --- baseline and reference configuration ---
    "M0":  ((32, 64),            {1, 2},    True),   # 2 conv blocks, 2 ECA
    "M1":  ((32, 64, 128),       {1, 2},    True),   # reference configuration
    "M2":  ((32, 64, 128, 256),  {1, 2},    True),   # depth +1
    "M3":  ((32, 64, 128),       {1, 2},    False),  # no Transformer (Flatten)
    # --- channel-attention gradient ---
    "M4":  ((32, 64, 128),       set(),     True),   # 0 ECA
    "M5":  ((32, 64, 128),       {1},       True),   # 1 ECA, block 1
    "M5b": ((32, 64, 128),       {2},       True),   # 1 ECA, block 2 (mirror)
    "M6":  ((32, 64, 128),       {1, 2, 3}, True),   # 3 ECA
}


# ======================================================================
# CLI: parameter counts for the model-complexity table
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build ablation variants and report parameter counts.")
    parser.add_argument("--size", type=int, default=9,
                        help="input grid side (9 for HIKARI, 6 for CIRA)")
    parser.add_argument("--only", default=None,
                        help="comma-separated variant names (default: all)")
    args = parser.parse_args()

    shape = (args.size, args.size, 1)
    names = args.only.split(",") if args.only else list(VARIANTS)

    print(f"{'variant':<8} {'params':>12} {'size_MB':>9}   spec")
    for name in names:
        conv, eca, tf_flag = VARIANTS[name]
        model = build_variant(shape, conv, eca, tf_flag)
        n = model.count_params()
        print(f"{name:<8} {n:>12,} {n * 4 / 2**20:>9.2f}   "
              f"conv={conv}, ECA after {sorted(eca) or '-'}, "
              f"transformer={tf_flag}")
