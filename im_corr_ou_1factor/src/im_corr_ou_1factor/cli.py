"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Calibrate correlated one-factor OU carry models")
    result.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
    result.add_argument("--refresh", action="store_true", help="Refresh the AkShare raw-data cache")
    return result


def main() -> None:
    args = parser().parse_args()
    config = load_config(Path(args.config))
    if args.refresh:
        config["data"]["refresh"] = True
    print(json.dumps(run_pipeline(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

