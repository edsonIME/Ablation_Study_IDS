# How Much Architecture Is Enough? A Controlled Ablation of Hybrid CNN–Attention IDS Models for HTTPS Traffic

> **Anonymous repository — under double-blind review.**
> All author-identifying information has been removed.

Code for a controlled, statistically grounded ablation study of hybrid
CNN–ECA–Transformer architectures for intrusion detection over encrypted
(HTTPS) flow metadata, evaluated on **HIKARI-2021** and
**CIRA-CIC-DoHBrw-2020** under a leakage-controlled protocol.

## Architectural variants

All variants are built from a single specification in `model_variants.py`
(so the trained models and the described models cannot diverge) and share the
same training protocol, seeds and data partitions; each changes exactly one
factor.

| Model | Conv. blocks | ECA modules (placement) | Transformer | Ablated factor |
|-------|--------------|-------------------------|-------------|----------------|
| M0    | 2 (32–64)          | 2 (blocks 1, 2) | yes | — (shallow baseline) |
| M1    | 3 (32–64–128)      | 2 (blocks 1, 2) | yes | — (reference configuration) |
| M2    | 4 (32–64–128–256)  | 2 (blocks 1, 2) | yes | depth (+1 block) |
| M3    | 3 (32–64–128)      | 2 (blocks 1, 2) | no  | global attention (Flatten head, dimensionally paired) |
| M4    | 3 (32–64–128)      | 0               | yes | channel attention (full removal) |
| M5    | 3 (32–64–128)      | 1 (block 1)     | yes | ECA dose 1 |
| M5b   | 3 (32–64–128)      | 1 (block 2)     | yes | ECA dose 1, mirrored placement |
| M6    | 3 (32–64–128)      | 3 (blocks 1–3)  | yes | ECA dose 3 (saturation) |

M4 → M5/M5b → M1 → M6 forms a channel-attention **dose-response gradient**
(0, 1, 1, 2, 3 modules); the mirrored pair M5/M5b disentangles module count
from placement. In the dose-response *slope*, dose 1 is the per-run mean of
M5 and M5b so it is not double-counted (`extended_analysis.py`).

### Channel attention (ECA)

`eca_block` implements Efficient Channel Attention: global average pooling
yields one descriptor per channel, a 1-D convolution over the channel
sequence (adaptive **odd** kernel) learns local cross-channel interactions,
and a sigmoid produces **one weight per channel** that recalibrates each
channel independently. `tests/test_eca_shape.py` asserts the gate varies
across channels (i.e. it is genuine per-channel attention, not a single
scalar gate).

## Repository layout

| File | Purpose |
|------|---------|
| `ablation_runner.py`      | Training driver: builds every variant from `model_variants.py`, trains with Monte Carlo Cross-Validation (resumable), and delegates the analysis to `statistical_analysis.py`. |
| `statistical_analysis.py` | **Canonical** statistics (single source of truth): auto-locates the per-run CSVs, pairs strictly by seed, runs Nadeau–Bengio-corrected paired tests, Holm–Bonferroni, effect sizes and percentage-point differences with confidence intervals, the dose-response table, and ready-to-paste LaTeX. Importable and used by the runner's `--stats-only`. |
| `extended_analysis.py`    | Auxiliary analyses built on the canonical primitives: convergence-failure rates, median/IQR, ECA dose-response slope, validation-tuned threshold helper, and model-complexity (params/FLOPs) helpers. |
| `model_variants.py`       | Single source of truth for the architectures: `eca_block`, `transformer_encoder`, `build_variant`, and the `VARIANTS` registry. |
| `tests/`                  | Analysis + split tests that need no TensorFlow, an optional ECA runtime test, and a synthetic results generator. |
| `requirements.txt`        | Python dependencies (tested with Python 3.10). |
| `LICENSE`                 | MIT. |

## Setup

Tested with **Python 3.10**. Dependencies are split so the analysis can be
run without TensorFlow:

```bash
pip install -r requirements-analysis.txt   # analysis + tests (no TensorFlow)
pip install -r requirements-training.txt   # adds GPU TensorFlow for training
# requirements.txt is a convenience alias for the training set.
```

Install the training set in its own environment (TensorFlow < 2.20 constrains
numpy < 2.1).

### Reproducibility

`requirements-lock.txt` is a full freeze (with transitive deps) of the analysis stack validated here.
The **training** environment must be frozen on the machine that runs the
experiments — fill `requirements-training-lock.txt`
(`python -m pip freeze > requirements-training-lock.txt`; review the output and remove any `file:///...`, `-e /path` or `user@host` references, which leak usernames/paths) and record the system
stack below (every result CSV also stores these versions inside its
`protocol_hash`):

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| TensorFlow | _fill from training env_ |
| scikit-learn | _fill from training env_ |
| imbalanced-learn | _fill from training env_ |
| numpy / pandas / scipy | _fill from training env_ |
| CUDA / cuDNN / NVIDIA driver | _fill from training env_ |
| GPU | _fill from training env_ |

## Quick functional check (no dataset, no TensorFlow)

Validate the whole analysis pipeline end to end using a synthetic results
folder — this does **not** reproduce any metric, it only checks the code
runs:

```bash
python tests/generate_synthetic_results.py --out results_demo
python statistical_analysis.py --results-dir results_demo --extra-contrast M5:M5b
python -m pytest tests/            # runs the full suite (recommended)
```

The analysis and split tests (`test_statistical_analysis.py`, `test_splitting.py`) run **without** TensorFlow. The optional `test_eca_shape.py` **requires** TensorFlow and is reported as *skipped* when absent (e.g. `... 1 skipped`), never as passed.

## Datasets

Download and place the CSVs in the repository root (override the filenames
with `--hikari-csv` / `--cira-csv`):

* **HIKARI-2021** (`ALLFLOWMETER_HIKARI2021.csv`) — https://zenodo.org/records/5199540
* **CIRA-CIC-DoHBrw-2020** (`CIRA.csv`) — https://www.unb.ca/cic/datasets/dohbrw-2020.html
  (only the layer-2 task, Benign vs. Malicious, is kept and mapped to 0/1)

**Leakage control.** Strong identifiers (IPs, ports, uid, timestamps) are
removed **by name** at load time. After their removal, exact duplicates can
remain in feature space; the **default** split is therefore **group-aware and
row-balanced**: identical feature vectors are hashed into a group, each group
is assigned as a whole to exactly one of train/val/test (so a duplicate can
never cross the partition), and groups are allocated greedily within each
class to approximate the **70/15/15 row fractions and the class ratio** — not
merely 70/15/15 of the group count. The **actual** partition sizes are
recorded per run (`n_train`, `n_val`, `n_test`) and drive the Nadeau–Bengio
ratio downstream. Exact (feature+label) duplicate rows are removed before splitting; remaining identical feature vectors are kept together by the group split, and with `--dedup-features` any conflicting-label group is dropped whole (never resolved by picking the first label). MinMax scaling is fit on the training rows and SMOTE is then applied to the scaled training set only. Use `--split random` only as a leakage diagnostic (never for reported
numbers) and `--dedup-features` to drop duplicates outright instead.

**Genuine Monte Carlo CV.** Each seed produces a **different** group-aware
partition (the allocation explores several seeded group orderings and keeps
the best-balanced one), so the 30 runs are genuine Monte Carlo resamples, not
30 copies of one split; the same seed always reproduces the same partition. A
per-seed `partition_hash` is recorded and the analysis **aborts** if the two
models of a contrast did not use the identical partition.

**Configuration stamping and protocol safety.** Every per-run row records the
resolved protocol — real partition sizes, hyper-parameters, split mode, the
architecture version, the code version and the key library versions — as a
`protocol_hash` (plus `model_hash`, `dataset_hash`). On resume the runner
**refuses to append** runs under a different protocol (or a file with no
protocol hash). The analysis **aborts** if a dataset's models were trained
under different protocols, if a file mixes protocols within a dataset, or if
runs are duplicated (within OR across files) — so incompatible runs are
never compared. By default the analysis also **requires** the provenance
columns and refuses files that lack them; pass `--allow-legacy-results` to
analyse older CSVs without these guarantees. Pairing is by **seed**.

## Usage

```bash
# 1) print the execution plan and exit (trains nothing; no TensorFlow needed)
python3 ablation_runner.py --dry-run

# 2) functional smoke test on a small subsample (needs the CIRA CSV present)
#    --max-samples is for a functional check only; --no-stats skips the final
#    analysis (which needs M1 + >=5 shared seeds and would otherwise error)
#    IMPORTANT: write to a SEPARATE results dir so the reduced-protocol smoke
#    output never mixes with (and later blocks) the full campaign in results/
python3 ablation_runner.py --models M5b --datasets CIRA \
    --n-runs 1 --epochs 2 --max-samples 5000 --no-stats \
    --results-dir results_smoke

# 3) full study, in the background (resumable: relaunch to continue)
#    Before training, the runner runs a PREFLIGHT that validates every
#    existing results/*.csv for the plan, so an incompatible file aborts in
#    seconds rather than after days.
nohup python3 ablation_runner.py --cira-csv CIRA.csv > ablation.log 2>&1 &
tail -f ablation.log

#    ...or run it in stages (same seed schedule and protocol; priority order runs
#    the most informative variants first). A stage without M1 finishes cleanly
#    and just skips the premature statistics (add --no-stats to silence them).
python3 ablation_runner.py --models M4 M3 M5 M5b M6
python3 ablation_runner.py --models M0 M1 M2

# 4) recompute statistics + LaTeX from the stored per-run CSVs
#    (exits non-zero if no results could be analysed, for CI/pipelines;
#    add --require-complete to abort unless all 30 seeds are present)
python3 ablation_runner.py --stats-only --require-complete
#    ...which is equivalent to calling the canonical analysis directly, once
#    per metric (the runner writes both fixed-0.5 and calibrated-t* tables):
python3 statistical_analysis.py --results-dir results --metric test_F1 \
    --extra-contrast M5:M5b --save-latex statistical_tables.tex
#    ...add --require-complete for the FINAL tables to abort unless every
#    model/dataset has all 30 seeds (42..71):
python3 statistical_analysis.py --results-dir results --metric test_F1 \
    --extra-contrast M5:M5b --require-complete --save-latex statistical_tables.tex
python3 statistical_analysis.py --results-dir results --metric tstar_F1 \
    --extra-contrast M5:M5b --require-complete \
    --save-latex statistical_tables_tstar.tex

# 5) auxiliary statistics (failure rates, ECA dose-response slope, complexity)
#    n_train/n_test are read from the CSVs; pass --n-train/--n-test only for
#    older CSVs that lack the recorded sizes
python3 extended_analysis.py --results-dir results

# 6) parameter counts per variant (input grid: 9 for HIKARI, 6 for CIRA)
python3 model_variants.py --size 9
```

**Runtime note:** a complete study (16 model–dataset pairs × 30 runs each)
takes several GPU-days on a mid-range GPU. Stage-wise execution via `--models`
is recommended; interrupting and relaunching is always safe. Staging preserves the same seed schedule and protocol; exact bit-for-bit floats are not guaranteed across GPU/driver/TensorFlow builds.

## Outputs

* `results/<MODEL>_<DATASET>.csv` — one row per Monte Carlo run (metrics at
  the fixed 0.5 threshold **and** at the validation-calibrated threshold t*,
  plus times, parameters and seeds).
* `results/feature_order_<DATASET>.json` — the exact feature (column) order that induces the 2D grid, written once per loaded dataset; its `dataset_hash` is stamped into every run so the order cannot drift unnoticed.
* `statistical_tables.tex` — ready-to-paste LaTeX (fixed-threshold protocol).
* `statistical_tables_tstar.tex` — same, at the validation-calibrated t*.

## Statistical protocol

`statistical_analysis.py` is the **single** implementation of the paired
protocol; `extended_analysis.py` calls its `compare_pair`, so the two never
diverge. Contrasts are paired strictly by **seed** over the Monte Carlo runs. The **Nadeau–Bengio-corrected paired t-test is the pre-specified
PRIMARY test for every contrast** (correction factor 1/N + n_test/n_train,
with n_test/n_train taken from the **mean recorded sizes over the shared seeds** of each contrast);
it is not chosen post hoc from a normality test. The **Wilcoxon** signed-rank
test is reported as an **uncorrected sensitivity** check (no canonical overlap
correction exists), never as a replacement; Shapiro–Wilk is a descriptive
diagnostic only. Holm–Bonferroni controls the family-wise error within each
dataset; the primary effect size is Cohen's d_z with a bootstrap CI, and the
mean-difference CI uses the **same** NB-corrected standard error as the test.


## License

MIT (see `LICENSE`). Released anonymously for double-blind review; the
copyright holder will be de-anonymized upon acceptance.
