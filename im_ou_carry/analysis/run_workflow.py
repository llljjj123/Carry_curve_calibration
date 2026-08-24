"""Clearly executable complete-workflow script for interactive analysis."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from im_ou_carry.config import load_config  # noqa: E402
from im_ou_carry.pipeline import run_pipeline  # noqa: E402


if __name__ == "__main__":
    result = run_pipeline(load_config(PROJECT_ROOT / "config.yaml"))
    for key, value in result.items():
        print(f"{key}: {value}")

