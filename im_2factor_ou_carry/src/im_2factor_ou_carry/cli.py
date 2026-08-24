"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Estimate the CSI 1000 / IM two-factor OU implied-carry term structure")
    result.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
    result.add_argument("--refresh", action="store_true", help="Refresh AkShare raw-data cache")
    return result


def main() -> None:
    args = parser().parse_args()
    config = load_config(Path(args.config))
    if args.refresh:
        config["data"]["refresh"] = True
    summary = run_pipeline(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
