"""Executable full-workflow script for interactive analysis."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from im_corr_ou_1factor.config import load_config  # noqa: E402
from im_corr_ou_1factor.pipeline import run_pipeline  # noqa: E402


if __name__ == "__main__":
    result = run_pipeline(load_config(PROJECT_ROOT / "config.yaml"))
    for key, value in result.items():
        print(f"{key}: {value}")

