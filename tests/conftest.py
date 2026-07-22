"""Test speed-ups: a smaller bootstrap and a session-scoped synthetic results
directory. Production code keeps the full 10,000-resample bootstrap; only the
unit tests use the reduced count."""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import statistical_analysis as sa   # noqa: E402


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch):
    orig = sa.bootstrap_ci
    monkeypatch.setattr(sa, "bootstrap_ci",
                        lambda v, fn, **k: orig(v, fn, **{**k, "n_boot": 200}))


@pytest.fixture(scope="session")
def synth_results(tmp_path_factory):
    """Build the 16 synthetic CSVs once per session (read-only)."""
    d = tmp_path_factory.mktemp("synth")
    subprocess.check_call([sys.executable,
                           str(REPO / "tests" / "generate_synthetic_results.py"),
                           "--out", str(d), "--n-runs", "30"])
    return d
