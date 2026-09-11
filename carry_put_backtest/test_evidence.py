"""Run pytest explicitly and persist auditable, source-bound evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def source_identity(root: Path) -> dict[str, str]:
    paths = sorted((root / "carry_put_backtest").glob("*.py"))
    paths += sorted((root / "carry_put_backtest/tests").glob("*.py"))
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def run_test_evidence(output: Path, command: list[str], *, root: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    process = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    ended = datetime.now(timezone.utc)
    match = re.search(r"(?:^|\s)(\d+) passed(?:[,\s]|$)", process.stdout)
    record = {
        "status": "passed" if process.returncode == 0 else "failed",
        "exit_code": process.returncode,
        "passed_count": int(match.group(1)) if match else None,
        "command": command,
        "interpreter": sys.executable,
        "started_at_utc": started.isoformat(),
        "ended_at_utc": ended.isoformat(),
        "source_identity": source_identity(root),
        "full_log_file": "test_results.log",
    }
    (output / "test_results.log").write_text(process.stdout, encoding="utf-8")
    (output / "test_results.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def classify_test_evidence(output: Path, *, root: Path) -> dict:
    path = output / "test_results.json"
    if not path.exists():
        return {"status": "not_run", "reason": "test_results.json is absent"}
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("source_identity") != source_identity(root):
        return {**record, "status": "stale", "reason": "source identity differs from test execution"}
    if record.get("exit_code") == 0 and record.get("status") == "passed":
        return record
    return {**record, "status": "failed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    pytest_args = args.pytest_args or [
        "carry_put_backtest/tests", "carry_put_pricing/tests", "Demo/tests",
        "im_2factor_ou_carry/tests", "-q", "-p", "no:cacheprovider",
    ]
    command = [sys.executable, "-B", "-m", "pytest", *pytest_args]
    result = run_test_evidence(args.output_dir, command, root=root)
    raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
