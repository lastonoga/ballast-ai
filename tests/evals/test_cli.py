from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_help_lists_dataset_from_traces():
    proc = subprocess.run(
        [sys.executable, "-m", "pydantic_ai_stateflow.evals.cli", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0
    assert (
        "dataset-from-traces" in proc.stdout.lower()
        or "dataset_from_traces" in proc.stdout.lower()
    )


def test_cli_dataset_from_traces_writes_yaml(tmp_path: Path):
    out = tmp_path / "ds.yaml"
    proc = subprocess.run(
        [
            sys.executable, "-m", "pydantic_ai_stateflow.evals.cli",
            "dataset-from-traces",
            "--pattern", "reflection",
            "--since", "2026-01-01",
            "--out", str(out),
            "--source", "demo",
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    text = out.read_text()
    assert "name:" in text
    assert "cases:" in text
