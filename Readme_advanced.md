# Readme_advanced

## Advanced execution guide for the anonymous CNN–ECA–Transformer ablation artifact

This document is a command-oriented companion to `README.md`. It covers installation, CPU/GPU validation, smoke tests, the complete 480-run study, staged execution, safe resumption, monitoring, integrity checks, statistics, auxiliary analyses, backups, cleanup, and troubleshooting.

All paths, usernames, hostnames, and environment names in this guide are generic. Run the commands from the repository root and replace only explicit placeholders such as `/path/to/anonymous_ablation_artifact` when necessary.

The commands are written for Linux/Ubuntu and are designed to be copied and pasted with minimal editing.

> **Important:** commands that use `--max-samples` or very few epochs are functional tests only. Never mix smoke-test results with the final experiment.

---

## 1. Repository and experiment overview

The complete study contains:

- 8 model variants: `M0`, `M1`, `M2`, `M3`, `M4`, `M5`, `M5b`, and `M6`;
- 2 datasets: `HIKARI` and `CIRA`;
- 30 Monte Carlo runs per model–dataset pair;
- seeds 42 through 71;
- up to 100 epochs per run.

Therefore, the final campaign contains:

```text
8 models × 2 datasets × 30 runs = 480 training runs
```

The default final protocol uses:

```text
split mode: group
training / validation / test targets: 70% / 15% / 15%
batch size: 64
maximum epochs: 100
base seed: 42
number of runs: 30
```

The exact hyperparameters are defined in `config.py`.

---

## 2. Open the project and define reusable paths

Run this from a terminal.

```bash
# Run this command from the repository root.
cd "/path/to/anonymous_ablation_artifact"
```

Define the dataset paths once. Replace the values below if the CSVs are stored elsewhere.

```bash
# Run from the repository root.
export PROJECT_DIR="$(pwd)"
export HIKARI_CSV="$PROJECT_DIR/ALLFLOWMETER_HIKARI2021.csv"
export CIRA_CSV="$PROJECT_DIR/CIRA-CIC-DoHBrw-2020.csv"
export FULL_RESULTS_DIR="$PROJECT_DIR/results_full"
export FULL_LOG="$PROJECT_DIR/ablation_full.log"
export FULL_PID="$PROJECT_DIR/ablation_full.pid"

cd "$PROJECT_DIR"
```

Confirm the paths:

```bash
printf 'Project: %s\nHIKARI:  %s\nCIRA:    %s\n' \
  "$PROJECT_DIR" "$HIKARI_CSV" "$CIRA_CSV"

ls -lh "$HIKARI_CSV" "$CIRA_CSV"
```

If either file is missing, correct the corresponding environment variable before continuing.

---

## 3. Create and activate a Python environment

### 3.1 Recommended training environment

```bash
python3 -m venv .venv-training
source .venv-training/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-training.txt
```

Confirm the active interpreter:

```bash
which python
python --version
python -m pip --version
```

### 3.2 Analysis-only environment without TensorFlow

Use this only when training is not required.

```bash
python3 -m venv .venv-analysis
source .venv-analysis/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-analysis.txt
```

### 3.3 Install from the exact training lock

After `requirements-training-lock.txt` has been generated from the final experimental environment:

```bash
python3 -m venv .venv-locked
source .venv-locked/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-training-lock.txt
```

---

## 4. Inspect the installed environment

```bash
python - <<'PY'
import platform
import sys

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())

packages = [
    "tensorflow",
    "keras",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "imblearn",
]

for name in packages:
    try:
        module = __import__(name)
        print(f"{name:12s}: {getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name:12s}: unavailable ({exc})")
PY
```

Check TensorFlow devices:

```bash
python - <<'PY'
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("CPUs:", tf.config.list_physical_devices("CPU"))
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY
```

Check NVIDIA availability when applicable:

```bash
nvidia-smi
```

If `nvidia-smi` is unavailable or reports no GPU, CPU execution is still supported.

---

## 5. Show every command-line option

```bash
python ablation_runner.py --help
python statistical_analysis.py --help
python extended_analysis.py --help
```

---

## 6. Compile the Python files

```bash
python -m py_compile \
  ablation_runner.py \
  config.py \
  model_variants.py \
  statistical_analysis.py \
  extended_analysis.py \
  tests/*.py
```

No output means compilation succeeded.

---

## 7. Run the complete test suite

### 7.1 Force CPU execution

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/
```

For the current validated revision, the expected result is:

```text
52 passed
```

### 7.2 Verbose test output

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -v tests/
```

### 7.3 Run only the ECA runtime test

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest \
  -q tests/test_eca_shape.py
```

### 7.4 Run only split tests

```bash
python -m pytest -q tests/test_splitting.py
```

### 7.5 Run only statistical-integrity tests

```bash
python -m pytest -q tests/test_statistical_analysis.py
```

### 7.6 Stop after the first failure

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest \
  -x -vv tests/
```

---

## 8. Build every model without training

### 8.1 HIKARI grid: 9 × 9

```bash
CUDA_VISIBLE_DEVICES=-1 python model_variants.py --size 9
```

### 8.2 CIRA grid: 6 × 6

```bash
CUDA_VISIBLE_DEVICES=-1 python model_variants.py --size 6
```

These commands validate model construction and print parameter counts.

Expected variants:

```text
M0 M1 M2 M3 M4 M5 M5b M6
```

A CIRA model that contains the Transformer may produce a warning about softmax over an axis of size one. This is expected because the 6 × 6 input pools down to one token.

---

## 9. Print the execution plan without training

This command does not require TensorFlow.

```bash
python ablation_runner.py \
  --hikari-csv "$HIKARI_CSV" \
  --cira-csv "$CIRA_CSV" \
  --datasets HIKARI CIRA \
  --n-runs 30 \
  --epochs 100 \
  --batch-size 64 \
  --split group \
  --results-dir "$FULL_RESULTS_DIR" \
  --dry-run
```

The output should list 16 model–dataset pairs.

---

## 10. Functional smoke tests

Always use a dedicated smoke-test directory. Never write reduced-protocol results into the final results directory.

### 10.1 CIRA smoke test on CPU

```bash
rm -rf "$PROJECT_DIR/results_smoke_cira"

CUDA_VISIBLE_DEVICES=-1 python -u ablation_runner.py \
  --cira-csv "$CIRA_CSV" \
  --models M5b \
  --datasets CIRA \
  --n-runs 1 \
  --epochs 1 \
  --batch-size 32 \
  --max-samples 1000 \
  --results-dir "$PROJECT_DIR/results_smoke_cira" \
  --no-stats
```

This validates:

- CIRA loading and label filtering;
- identifier removal;
- group split;
- scaling;
- SMOTE;
- ECA;
- Transformer execution with one token;
- threshold calibration;
- metric calculation;
- CSV writing.

### 10.2 HIKARI smoke test on CPU

```bash
rm -rf "$PROJECT_DIR/results_smoke_hikari"

CUDA_VISIBLE_DEVICES=-1 python -u ablation_runner.py \
  --hikari-csv "$HIKARI_CSV" \
  --models M1 M3 \
  --datasets HIKARI \
  --n-runs 1 \
  --epochs 1 \
  --batch-size 32 \
  --max-samples 5000 \
  --results-dir "$PROJECT_DIR/results_smoke_hikari" \
  --no-stats
```

This validates:

- the 9 × 9 input grid;
- four Transformer tokens;
- M1 with Transformer;
- M3 without Transformer;
- identical partition hashes for paired variants using the same seed.

### 10.3 CIRA smoke test on GPU

```bash
rm -rf "$PROJECT_DIR/results_smoke_cira_gpu"

CUDA_VISIBLE_DEVICES=0 python -u ablation_runner.py \
  --cira-csv "$CIRA_CSV" \
  --models M5b \
  --datasets CIRA \
  --n-runs 1 \
  --epochs 1 \
  --batch-size 32 \
  --max-samples 1000 \
  --results-dir "$PROJECT_DIR/results_smoke_cira_gpu" \
  --no-stats
```

### 10.4 HIKARI smoke test on GPU

```bash
rm -rf "$PROJECT_DIR/results_smoke_hikari_gpu"

CUDA_VISIBLE_DEVICES=0 python -u ablation_runner.py \
  --hikari-csv "$HIKARI_CSV" \
  --models M1 M3 \
  --datasets HIKARI \
  --n-runs 1 \
  --epochs 1 \
  --batch-size 32 \
  --max-samples 5000 \
  --results-dir "$PROJECT_DIR/results_smoke_hikari_gpu" \
  --no-stats
```

---

## 11. Test safe resumption

First create two CIRA smoke-test runs:

```bash
rm -rf "$PROJECT_DIR/results_smoke_resume"

CUDA_VISIBLE_DEVICES=-1 python -u ablation_runner.py \
  --cira-csv "$CIRA_CSV" \
  --models M5b \
  --datasets CIRA \
  --n-runs 2 \
  --epochs 1 \
  --batch-size 32 \
  --max-samples 1000 \
  --results-dir "$PROJECT_DIR/results_smoke_resume" \
  --no-stats
```

Run exactly the same command again:

```bash
CUDA_VISIBLE_DEVICES=-1 python -u ablation_runner.py \
  --cira-csv "$CIRA_CSV" \
  --models M5b \
  --datasets CIRA \
  --n-runs 2 \
  --epochs 1 \
  --batch-size 32 \
  --max-samples 1000 \
  --results-dir "$PROJECT_DIR/results_smoke_resume" \
  --no-stats
```

Expected message:

```text
already complete (2/2); skipping
```

Inspect the CSV:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("results_smoke_resume/M5b_CIRA.csv")
df = pd.read_csv(path)

print(df[[
    "run",
    "seed",
    "test_F1",
    "tstar_F1",
    "partition_hash",
]])

print("\nRows:", len(df))
print("Seeds:", df["seed"].astype(int).tolist())
print("Duplicate seeds:", bool(df["seed"].duplicated().any()))
print("Unique partitions:", df["partition_hash"].nunique())
PY
```

Expected:

```text
Rows: 2
Seeds: [42, 43]
Duplicate seeds: False
Unique partitions: 2
```

---

## 12. Prepare the final campaign directory

Do not reuse a smoke-test directory.

```bash
mkdir -p "$FULL_RESULTS_DIR"
```

Inspect it:

```bash
find "$FULL_RESULTS_DIR" \
  -maxdepth 1 \
  -type f \
  -printf '%f\n' \
  | sort
```

For a new campaign, the directory should contain no CSV files.

To deliberately start over, first make a backup and then remove the directory:

```bash
if [ -d "$FULL_RESULTS_DIR" ]; then
  tar -czf "results_full_before_reset_$(date +%Y%m%d_%H%M%S).tar.gz" \
    "$FULL_RESULTS_DIR"
fi

rm -rf "$FULL_RESULTS_DIR"
mkdir -p "$FULL_RESULTS_DIR"
```

Never reset a directory that contains runs you intend to preserve.

---

## 13. Run the complete study in the foreground

### 13.1 CPU-only complete campaign

```bash
CUDA_VISIBLE_DEVICES=-1 python -u ablation_runner.py \
  --hikari-csv "$HIKARI_CSV" \
  --cira-csv "$CIRA_CSV" \
  --datasets HIKARI CIRA \
  --n-runs 30 \
  --epochs 100 \
  --batch-size 64 \
  --split group \
  --results-dir "$FULL_RESULTS_DIR" \
  --no-stats
```

### 13.2 Single-GPU complete campaign

```bash
CUDA_VISIBLE_DEVICES=0 python -u ablation_runner.py \
  --hikari-csv "$HIKARI_CSV" \
  --cira-csv "$CIRA_CSV" \
  --datasets HIKARI CIRA \
  --n-runs 30 \
  --epochs 100 \
  --batch-size 64 \
  --split group \
  --results-dir "$FULL_RESULTS_DIR" \
  --no-stats
```

A CPU campaign is valid but may require days or weeks. A GPU is strongly recommended for the complete study.

---

## 14. Run the complete study in the background

### 14.1 CPU-only background execution

```bash
nohup env \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES=-1 \
  python -u ablation_runner.py \
  --hikari-csv "$HIKARI_CSV" \
  --cira-csv "$CIRA_CSV" \
  --datasets HIKARI CIRA \
  --n-runs 30 \
  --epochs 100 \
  --batch-size 64 \
  --split group \
  --results-dir "$FULL_RESULTS_DIR" \
  --no-stats \
  > "$FULL_LOG" 2>&1 &

echo $! | tee "$FULL_PID"
```

### 14.2 Single-GPU background execution

```bash
nohup env \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES=0 \
  python -u ablation_runner.py \
  --hikari-csv "$HIKARI_CSV" \
  --cira-csv "$CIRA_CSV" \
  --datasets HIKARI CIRA \
  --n-runs 30 \
  --epochs 100 \
  --batch-size 64 \
  --split group \
  --results-dir "$FULL_RESULTS_DIR" \
  --no-stats \
  > "$FULL_LOG" 2>&1 &

echo $! | tee "$FULL_PID"
```

---

## 15. Run the study in stages

All stages must use exactly the same:

- dataset files;
- split mode;
- batch size;
- maximum epochs;
- dependency versions;
- code version;
- results directory;
- `--max-samples` setting;
- deduplication setting.

### 15.1 High-priority ablations first

```bash
CUDA_VISIBLE_DEVICES=0 python -u ablation_runner.py \
  --hikari-csv "$HIKARI_CSV" \
  --cira-csv "$CIRA_CSV" \
  --models M4 M3 M5 M5b M6 \
  --datasets HIKARI CIRA \
  --n-runs 30 \
  --epochs 100 \
  --batch-size 64 \
  --split group \
  --results-dir "$FULL_RESULTS_DIR" \
  --no-stats
```

### 15.2 Reference and depth variants

```bash
CUDA_VISIBLE_DEVICES=0 python -u ablation_runner.py \
  --hikari-csv "$HIKARI_CSV" \
  --cira-csv "$CIRA_CSV" \
  --models M0 M1 M2 \
  --datasets HIKARI CIRA \
  --n-runs 30 \
  --epochs 100 \
  --batch-size 64 \
  --split group \
  --results-dir "$FULL_RESULTS_DIR" \
  --no-stats
```

### 15.3 Run one dataset at a time

HIKARI:

```bash
CUDA_VISIBLE_DEVICES=0 python -u ablation_runner.py \
  --hikari-csv "$HIKARI_CSV" \
  --models M0 M1 M2 M3 M4 M5 M5b M6 \
  --datasets HIKARI \
  --n-runs 30 \
  --epochs 100 \
  --batch-size 64 \
  --split group \
  --results-dir "$FULL_RESULTS_DIR" \
  --no-stats
```

CIRA:

```bash
CUDA_VISIBLE_DEVICES=0 python -u ablation_runner.py \
  --cira-csv "$CIRA_CSV" \
  --models M0 M1 M2 M3 M4 M5 M5b M6 \
  --datasets CIRA \
  --n-runs 30 \
  --epochs 100 \
  --batch-size 64 \
  --split group \
  --results-dir "$FULL_RESULTS_DIR" \
  --no-stats
```

Both dataset-specific commands must use the same protocol and software environment so their `study_protocol_hash` remains identical.

---

## 16. Monitor a background campaign

### 16.1 Follow the complete log

```bash
tail -f "$FULL_LOG"
```

Press `Ctrl+C` to leave `tail`. This does not stop the background process.

### 16.2 Show only completed runs and important events

```bash
grep --line-buffered -E \
  '\[[0-9]+/30\]|already complete|Training finished|Traceback|Killed|MemoryError|No space left' \
  "$FULL_LOG"
```

### 16.3 Check whether the process is running

```bash
ps -fp "$(cat "$FULL_PID")"
```

### 16.4 Show CPU and memory use for the process

```bash
top -p "$(cat "$FULL_PID")"
```

### 16.5 Interactive process monitoring

```bash
htop
```

### 16.6 GPU monitoring

```bash
watch -n 2 nvidia-smi
```

### 16.7 Check free memory and disk space

```bash
free -h
df -h "$PROJECT_DIR"
```

### 16.8 Search for fatal errors

```bash
grep -nE \
  'Traceback|Killed|MemoryError|No space left|SystemExit' \
  "$FULL_LOG"
```

CUDA initialization messages may appear on CPU-only machines and are not necessarily fatal. A Python traceback is the stronger indicator of failure.

---

## 17. Count completed runs

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("results_full")
total = 0

csv_files = sorted(root.glob("*.csv"))

if not csv_files:
    print("No completed runs have been written yet.")
else:
    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"ERROR {path.name}: {exc}")
            continue

        count = len(df)
        total += count
        print(f"{path.name:22s}: {count:2d}/30")

    print(f"\nTotal completed: {total}/480")
PY
```

Watch progress every 60 seconds:

```bash
watch -n 60 '
python - <<'"'"'PY'"'"'
from pathlib import Path
import pandas as pd

root = Path("results_full")
total = 0

for path in sorted(root.glob("*.csv")):
    try:
        count = len(pd.read_csv(path))
    except Exception:
        count = 0
    total += count
    print(f"{path.name:22s}: {count:2d}/30")

print(f"\nTotal completed: {total}/480")
PY
'
```

---

## 18. Stop the background campaign safely

Send a normal termination signal:

```bash
kill -TERM "$(cat "$FULL_PID")"
```

Wait briefly and check:

```bash
sleep 5
ps -fp "$(cat "$FULL_PID")"
```

Only use a forceful kill if the process does not stop:

```bash
kill -KILL "$(cat "$FULL_PID")"
```

The runner writes completed rows atomically. The currently active run may be lost, but previously completed runs should remain resumable.

---

## 19. Resume an interrupted campaign

Use exactly the same command and protocol. Append to the existing log with `>>`.

### 19.1 Resume on CPU

```bash
nohup env \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES=-1 \
  python -u ablation_runner.py \
  --hikari-csv "$HIKARI_CSV" \
  --cira-csv "$CIRA_CSV" \
  --datasets HIKARI CIRA \
  --n-runs 30 \
  --epochs 100 \
  --batch-size 64 \
  --split group \
  --results-dir "$FULL_RESULTS_DIR" \
  --no-stats \
  >> "$FULL_LOG" 2>&1 &

echo $! > "$FULL_PID"
```

### 19.2 Resume on GPU

```bash
nohup env \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES=0 \
  python -u ablation_runner.py \
  --hikari-csv "$HIKARI_CSV" \
  --cira-csv "$CIRA_CSV" \
  --datasets HIKARI CIRA \
  --n-runs 30 \
  --epochs 100 \
  --batch-size 64 \
  --split group \
  --results-dir "$FULL_RESULTS_DIR" \
  --no-stats \
  >> "$FULL_LOG" 2>&1 &

echo $! > "$FULL_PID"
```

Do not modify the code, dataset contents, package versions, epochs, batch size, split mode, or deduplication policy between resumptions.

---

## 20. Verify all 480 runs before final statistics

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("results_full")
models = ["M0", "M1", "M2", "M3", "M4", "M5", "M5b", "M6"]
datasets = ["HIKARI", "CIRA"]
expected_seeds = set(range(42, 72))

ok = True
total = 0

for dataset in datasets:
    for model in models:
        path = root / f"{model}_{dataset}.csv"

        if not path.exists():
            print(f"MISSING {path}")
            ok = False
            continue

        df = pd.read_csv(path)
        seeds = set(pd.to_numeric(df["seed"], errors="raise").astype(int))
        missing = sorted(expected_seeds - seeds)
        extra = sorted(seeds - expected_seeds)
        duplicated = bool(df["seed"].duplicated().any())
        unique_partitions = df["partition_hash"].nunique()

        valid = (
            len(df) == 30
            and not missing
            and not extra
            and not duplicated
            and unique_partitions == 30
        )

        status = "OK" if valid else "ERROR"
        print(
            f"{status:5s} {path.name:22s} "
            f"rows={len(df):2d} "
            f"missing={missing} "
            f"extra={extra} "
            f"duplicate_seed={duplicated} "
            f"partitions={unique_partitions}"
        )

        total += len(df)
        ok &= valid

print(f"\nTotal: {total}/480")

if not ok:
    raise SystemExit("The campaign is incomplete or inconsistent.")

print("Complete 480-run campaign confirmed.")
PY
```

---

## 21. Run the final strict analysis through the runner

```bash
python ablation_runner.py \
  --stats-only \
  --results-dir "$FULL_RESULTS_DIR" \
  --require-complete
```

This generates:

```text
statistical_tables.tex
statistical_tables_tstar.tex
```

The command exits with a nonzero status if the study is incomplete or inconsistent.

---

## 22. Run the canonical statistical analysis directly

### 22.1 Fixed threshold F1

```bash
python statistical_analysis.py \
  --results-dir "$FULL_RESULTS_DIR" \
  --metric test_F1 \
  --reference M1 \
  --extra-contrast M5:M5b \
  --require-complete \
  --save-latex statistical_tables.tex
```

### 22.2 Validation-calibrated threshold F1

```bash
python statistical_analysis.py \
  --results-dir "$FULL_RESULTS_DIR" \
  --metric tstar_F1 \
  --reference M1 \
  --extra-contrast M5:M5b \
  --require-complete \
  --save-latex statistical_tables_tstar.tex
```

### 22.3 Print LaTeX to the terminal

```bash
python statistical_analysis.py \
  --results-dir "$FULL_RESULTS_DIR" \
  --metric test_F1 \
  --extra-contrast M5:M5b \
  --require-complete \
  --emit-latex
```

### 22.4 Analyze AUC

```bash
python statistical_analysis.py \
  --results-dir "$FULL_RESULTS_DIR" \
  --metric test_AUC \
  --reference M1 \
  --extra-contrast M5:M5b \
  --require-complete \
  --save-latex statistical_tables_auc.tex
```

### 22.5 Analyze precision

```bash
python statistical_analysis.py \
  --results-dir "$FULL_RESULTS_DIR" \
  --metric test_Prec \
  --reference M1 \
  --extra-contrast M5:M5b \
  --require-complete \
  --save-latex statistical_tables_precision.tex
```

### 22.6 Analyze recall

```bash
python statistical_analysis.py \
  --results-dir "$FULL_RESULTS_DIR" \
  --metric test_Rec \
  --reference M1 \
  --extra-contrast M5:M5b \
  --require-complete \
  --save-latex statistical_tables_recall.tex
```

### 22.7 Change the practical-equivalence margin

The default margin is 0.5 percentage points.

```bash
python statistical_analysis.py \
  --results-dir "$FULL_RESULTS_DIR" \
  --metric test_F1 \
  --equiv-margin 1.0 \
  --extra-contrast M5:M5b \
  --require-complete
```

### 22.8 Analyze one long-format CSV

```bash
python statistical_analysis.py \
  --csv merged_results.csv \
  --metric test_F1 \
  --extra-contrast M5:M5b \
  --require-complete
```

### 22.9 Legacy CSV analysis

Use only for old files that lack provenance columns. Integrity guarantees are reduced.

```bash
python statistical_analysis.py \
  --results-dir legacy_results \
  --metric test_F1 \
  --allow-legacy-results
```

---

## 23. Run auxiliary analyses

```bash
rm -rf "$PROJECT_DIR/stats_report_full"

python extended_analysis.py \
  --results-dir "$FULL_RESULTS_DIR" \
  --out "$PROJECT_DIR/stats_report_full"
```

Expected files:

```text
stats_report_full/descriptive.csv
stats_report_full/pairwise.csv
stats_report_full/eca_gradient.csv
stats_report_full/table_rows.tex
```

List them:

```bash
find "$PROJECT_DIR/stats_report_full" \
  -maxdepth 1 \
  -type f \
  -printf '%f\n' \
  | sort
```

For legacy result files that do not contain recorded split sizes:

```bash
python extended_analysis.py \
  --results-dir legacy_results \
  --out stats_report_legacy \
  --n-train 7000 \
  --n-test 1500 \
  --allow-legacy-results
```

---

## 24. Test the statistics with synthetic results

### 24.1 Complete synthetic study

```bash
rm -rf results_demo stats_report_demo

python tests/generate_synthetic_results.py \
  --out results_demo \
  --n-runs 30
```

Run the final strict statistics:

```bash
python ablation_runner.py \
  --stats-only \
  --results-dir results_demo \
  --require-complete
```

Run the auxiliary analysis:

```bash
python extended_analysis.py \
  --results-dir results_demo \
  --out stats_report_demo
```

### 24.2 Confirm that an incomplete study is rejected

```bash
rm -rf results_incomplete

python tests/generate_synthetic_results.py \
  --out results_incomplete \
  --n-runs 5
```

This command must fail:

```bash
python ablation_runner.py \
  --stats-only \
  --results-dir results_incomplete \
  --require-complete
```

A successful rejection confirms that final tables cannot be generated from only five seeds.

---

## 25. Diagnostic split modes

### 25.1 Leakage-safe default

```bash
python ablation_runner.py \
  --hikari-csv "$HIKARI_CSV" \
  --cira-csv "$CIRA_CSV" \
  --split group \
  --dry-run
```

Use `group` for all reported results.

### 25.2 Random split diagnostic

```bash
CUDA_VISIBLE_DEVICES=-1 python -u ablation_runner.py \
  --cira-csv "$CIRA_CSV" \
  --models M1 \
  --datasets CIRA \
  --n-runs 1 \
  --epochs 1 \
  --batch-size 32 \
  --max-samples 1000 \
  --split random \
  --results-dir results_random_diagnostic \
  --no-stats
```

This is a leakage diagnostic only. Do not report it as part of the final study.

### 25.3 Drop conflicting duplicate-feature groups

```bash
CUDA_VISIBLE_DEVICES=-1 python -u ablation_runner.py \
  --cira-csv "$CIRA_CSV" \
  --models M1 \
  --datasets CIRA \
  --n-runs 1 \
  --epochs 1 \
  --batch-size 32 \
  --max-samples 1000 \
  --split group \
  --dedup-features \
  --results-dir results_dedup_diagnostic \
  --no-stats
```

Do not enable this option for final results unless it is explicitly part of the paper protocol.

---

## 26. Inspect the generated provenance

List final CSV columns:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

path = next(Path("results_full").glob("*.csv"))
df = pd.read_csv(path)

print("File:", path)
print("\nColumns:")
for column in df.columns:
    print(" -", column)

print("\nFirst row:")
print(df.head(1).T)
PY
```

Inspect feature order:

```bash
python -m json.tool \
  "$FULL_RESULTS_DIR/feature_order_HIKARI.json" \
  | less
```

```bash
python -m json.tool \
  "$FULL_RESULTS_DIR/feature_order_CIRA.json" \
  | less
```

Compare study hashes across all files:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("results_full")

for path in sorted(root.glob("*.csv")):
    df = pd.read_csv(path)
    study = sorted(df["study_protocol_hash"].astype(str).unique())
    protocol = sorted(df["protocol_hash"].astype(str).unique())
    model_hash = sorted(df["model_hash"].astype(str).unique())

    print(
        f"{path.name:22s} "
        f"study={study} "
        f"protocol={protocol} "
        f"model={model_hash}"
    )
PY
```

Every file should contain one study hash, and the study hash should be identical across HIKARI and CIRA.

---

## 27. Freeze and sanitize the final training environment

Generate the lock on the machine used for the final experiments:

```bash
python -m pip freeze \
  > requirements-training-lock.txt
```

Search for personal or machine-specific references:

```bash
grep -nE \
  'file:|/home/|/Users/|^-e |git\+|user@|username|hostname' \
  requirements-training-lock.txt
```

Inspect direct references:

```bash
grep -n ' @ ' requirements-training-lock.txt
```

A standard PyPI lock should use lines such as:

```text
tensorflow==...
numpy==...
pandas==...
```

Remove or replace local path references before publication.

Save system information without usernames or hostnames:

```bash
{
  echo "Python: $(python --version 2>&1)"
  echo "OS: $(. /etc/os-release && echo "$PRETTY_NAME")"
  echo "Kernel: $(uname -r)"
  python - <<'PY'
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy
import sklearn
import imblearn

print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("SciPy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("imbalanced-learn:", imblearn.__version__)
PY
  nvidia-smi --query-gpu=name,driver_version \
    --format=csv,noheader 2>/dev/null || true
} > training_environment.txt
```

Review `training_environment.txt` before sharing it.

---

## 28. Back up the completed experiment

```bash
tar -czf "ablation_results_backup_$(date +%Y%m%d_%H%M%S).tar.gz" \
  "$FULL_RESULTS_DIR" \
  "$FULL_LOG" \
  statistical_tables.tex \
  statistical_tables_tstar.tex \
  stats_report_full
```

Generate a checksum:

```bash
sha256sum ablation_results_backup_*.tar.gz \
  > ablation_results_backup.sha256
```

Verify later:

```bash
sha256sum -c ablation_results_backup.sha256
```

Keep experimental backups outside the anonymous Git repository.

---

## 29. Clean generated files before publication

```bash
rm -rf \
  results \
  results_full \
  results_smoke \
  results_smoke_* \
  results_demo \
  results_incomplete \
  results_random_diagnostic \
  results_dedup_diagnostic \
  stats_report \
  stats_report_* \
  .pytest_cache \
  __pycache__ \
  tests/__pycache__
```

Remove generated tables and logs:

```bash
rm -f \
  *.log \
  *.pid \
  statistical_tables*.tex \
  table_rows.tex \
  analysis_report*.txt
```

Do not delete a completed experiment unless it has been safely backed up.

---

## 30. Check repository hygiene before committing

Find files that should normally not be published:

```bash
find . -type f \
  \( \
    -name "*.csv" -o \
    -name "*.log" -o \
    -name "*.pid" -o \
    -name "*.tex" -o \
    -name "*.npy" -o \
    -name "*.npz" -o \
    -name "*.pkl" -o \
    -name "*.h5" -o \
    -name "*.keras" -o \
    -name "*.zip" -o \
    -name "*.tar.gz" \
  \) \
  -print
```

Confirm ignored paths:

```bash
git check-ignore -v \
  results/M1_HIKARI.csv \
  results_full/M1_HIKARI.csv \
  results_smoke/M5b_CIRA.csv \
  results_smoke_hikari/M1_HIKARI.csv \
  ablation_full.log \
  statistical_tables.tex
```

Check repository status:

```bash
git status --short
```

Show staged files:

```bash
git diff --cached --name-only
```

Search tracked files for possible identifiers:

```bash
grep -RInE \
  'username|/home/|@.*\.(com|org|br)|ORCID|institution|university|institute' \
  --exclude-dir=.git \
  --exclude='requirements-training-lock.txt' \
  .
```

Review every match manually. Dataset titles and public URLs are expected; personal names, local paths, and private email addresses are not.

---

## 31. Useful troubleshooting commands

### 31.1 CUDA warning on a CPU-only machine

Messages such as these are expected when no CUDA driver is available:

```text
Could not find cuda drivers on your machine
GPU will not be used
failed call to cuInit
```

Confirm CPU execution:

```bash
CUDA_VISIBLE_DEVICES=-1 python - <<'PY'
import tensorflow as tf
print(tf.config.list_physical_devices())
PY
```

### 31.2 CIRA softmax warning

CIRA produces one token after pooling. Keras may report:

```text
softmax over an axis of size 1
```

This is expected. On CIRA, the M1–M3 contrast measures the contribution of the whole Transformer block rather than attention between multiple positions.

### 31.3 `± nan` after a smoke test

A standard deviation cannot be calculated from one run. Use at least two runs to obtain a finite sample standard deviation.

### 31.4 Protocol mismatch on resume

Do not edit the CSV manually. Verify that the resumed command uses exactly the same:

```text
dataset files
code
package versions
epochs
batch size
split
max-samples
deduplication setting
```

Use a new results directory if intentionally changing the protocol.

### 31.5 No completed CSV yet

The runner writes a row only after a run finishes. During the first long run, the results directory can remain empty.

### 31.6 Process was killed by the operating system

Check memory:

```bash
free -h
dmesg -T | tail -100
```

Search for out-of-memory events:

```bash
dmesg -T | grep -iE 'out of memory|oom|killed process'
```

### 31.7 Disk full

```bash
df -h
du -sh "$PROJECT_DIR"/* 2>/dev/null | sort -h
```

### 31.8 Confirm that the CSV is readable

```bash
python - <<'PY'
import pandas as pd

path = "results_full/M1_HIKARI.csv"
df = pd.read_csv(path)

print(df.shape)
print(df.tail())
PY
```

### 31.9 Show the last 100 log lines

```bash
tail -n 100 "$FULL_LOG"
```

### 31.10 Show only the last completed runs

```bash
grep -E '\[[0-9]+/30\]' "$FULL_LOG" \
  | tail -20
```

---

## 32. Command-line option reference

### `ablation_runner.py`

```text
--results-dir DIR
    Directory used for per-run CSVs and feature-order JSON files.

--hikari-csv FILE
    HIKARI CSV path.

--cira-csv FILE
    CIRA CSV path.

--models MODEL [MODEL ...]
    Subset of M0 M1 M2 M3 M4 M5 M5b M6.

--datasets DATASET [DATASET ...]
    HIKARI, CIRA, or both.

--n-runs N
    Number of Monte Carlo runs beginning at seed 42.

--epochs N
    Maximum training epochs per run.

--batch-size N
    Training batch size.

--split group|random
    Use group for final leakage-controlled results.
    Random is only a diagnostic.

--dedup-features
    Drop entire conflicting-label feature groups.
    Do not enable for final results unless specified by the paper protocol.

--max-samples N
    Cap the cleaned dataset.
    Functional smoke tests only.

--stats-only
    Skip training and analyze stored CSVs.

--require-complete
    With --stats-only, require all 30 seeds for every model and dataset.

--no-stats
    Train without automatically running statistics afterward.

--dry-run
    Print the execution plan and exit.
```

### `statistical_analysis.py`

```text
--results-dir DIR
    Read all canonical per-model CSVs from a directory.

--csv FILE
    Read one long-format CSV instead.

--metric COLUMN
    Metric column, such as test_F1, tstar_F1, test_AUC, test_Prec, or test_Rec.

--reference MODEL
    Reference model for default pairwise contrasts. Default: M1.

--train-frac FLOAT
--test-frac FLOAT
    Fallback fractions for legacy data lacking recorded sizes.

--equiv-margin FLOAT
    Practical-equivalence margin in percentage points. Default: 0.5.

--extra-contrast A:B
    Add a paired contrast, for example M5:M5b.
    The option may be repeated.

--emit-latex
    Print LaTeX to standard output.

--save-latex [FILE]
    Save LaTeX to a file.

--require-complete
    Require all 30 seeds for every model–dataset pair.

--allow-legacy-results
    Permit older CSVs without provenance fields.
    This reduces integrity guarantees.
```

### `extended_analysis.py`

```text
--results-dir DIR
    Required results directory.

--out DIR
    Output directory. Default: stats_report.

--n-train N
--n-test N
    Legacy fallback sizes when CSV rows do not record them.

--allow-legacy-results
    Permit legacy CSVs with reduced provenance guarantees.
```

---

## 33. Final publication checklist

```text
[ ] Dataset files are not committed.
[ ] Experimental results are not committed.
[ ] All tests pass with TensorFlow installed.
[ ] All eight variants build for input sizes 9 and 6.
[ ] CIRA CPU/GPU smoke test passes.
[ ] HIKARI CPU/GPU smoke test passes.
[ ] Resume test skips completed seeds.
[ ] Two different seeds have different partition hashes.
[ ] Synthetic complete study passes --require-complete.
[ ] Synthetic incomplete study is rejected.
[ ] Final training lock is populated and sanitized.
[ ] README environment table is populated.
[ ] The complete campaign has 480 valid rows.
[ ] Final F1 and calibrated-threshold tables are generated.
[ ] Auxiliary analysis outputs are generated.
[ ] The anonymous Git history contains no author identity or old personal commits.
```

---

## 34. Minimal production sequence

The following is the shortest safe production workflow.

### Start the complete study on GPU

```bash
# Run from the repository root.
export PROJECT_DIR="$(pwd)"
export HIKARI_CSV="$PROJECT_DIR/ALLFLOWMETER_HIKARI2021.csv"
export CIRA_CSV="$PROJECT_DIR/CIRA-CIC-DoHBrw-2020.csv"
export FULL_RESULTS_DIR="$PROJECT_DIR/results_full"
export FULL_LOG="$PROJECT_DIR/ablation_full.log"
export FULL_PID="$PROJECT_DIR/ablation_full.pid"

cd "$PROJECT_DIR"
source .venv-training/bin/activate

nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
  python -u ablation_runner.py \
  --hikari-csv "$HIKARI_CSV" \
  --cira-csv "$CIRA_CSV" \
  --datasets HIKARI CIRA \
  --n-runs 30 \
  --epochs 100 \
  --batch-size 64 \
  --split group \
  --results-dir "$FULL_RESULTS_DIR" \
  --no-stats \
  > "$FULL_LOG" 2>&1 &

echo $! | tee "$FULL_PID"
tail -f "$FULL_LOG"
```

### Start the complete study on CPU

```bash
# Run from the repository root.
export PROJECT_DIR="$(pwd)"
export HIKARI_CSV="$PROJECT_DIR/ALLFLOWMETER_HIKARI2021.csv"
export CIRA_CSV="$PROJECT_DIR/CIRA-CIC-DoHBrw-2020.csv"
export FULL_RESULTS_DIR="$PROJECT_DIR/results_full"
export FULL_LOG="$PROJECT_DIR/ablation_full.log"
export FULL_PID="$PROJECT_DIR/ablation_full.pid"

cd "$PROJECT_DIR"
source .venv-training/bin/activate

nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=-1 \
  python -u ablation_runner.py \
  --hikari-csv "$HIKARI_CSV" \
  --cira-csv "$CIRA_CSV" \
  --datasets HIKARI CIRA \
  --n-runs 30 \
  --epochs 100 \
  --batch-size 64 \
  --split group \
  --results-dir "$FULL_RESULTS_DIR" \
  --no-stats \
  > "$FULL_LOG" 2>&1 &

echo $! | tee "$FULL_PID"
tail -f "$FULL_LOG"
```

### Generate final analyses after all 480 runs

```bash
cd "$PROJECT_DIR"
source .venv-training/bin/activate

python ablation_runner.py \
  --stats-only \
  --results-dir "$FULL_RESULTS_DIR" \
  --require-complete

python extended_analysis.py \
  --results-dir "$FULL_RESULTS_DIR" \
  --out "$PROJECT_DIR/stats_report_full"
```
