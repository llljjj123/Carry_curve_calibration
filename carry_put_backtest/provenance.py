"""Immutable provenance helpers for historical back-test evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_tree(path: Path, *, root: Path) -> dict[str, str]:
    return {
        str(file.relative_to(root)).replace("\\", "/"): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def write_historical_evidence_audit(root: Path, destination: Path) -> dict:
    """Snapshot preserved outputs and compare checkpoint source identities."""
    preserved = (
        "carry_put_backtest/outputs_short_spot",
        "carry_put_backtest/outputs_historical_accelerated",
        "carry_put_backtest/outputs_historical",
    )
    raw_inputs = (
        "im_2factor_ou_carry/data/raw/spot_raw.csv",
        "im_2factor_ou_carry/data/raw/futures_raw.csv",
    )
    artifact_hashes: dict[str, str] = {}
    for relative in preserved:
        artifact_hashes.update(hash_tree(root / relative, root=root))
    input_hashes = {
        relative: sha256_file(root / relative)
        for relative in raw_inputs
        if (root / relative).exists()
    }

    source_names = (
        "carry_put_backtest/engine.py",
        "carry_put_backtest/analyze_historical.py",
        "carry_put_backtest/analyze_short_spot.py",
        "carry_put_backtest/reporting.py",
    )
    current = {name: sha256_file(root / name) for name in source_names}
    mismatches = []
    checkpoints = sorted((root / "carry_put_backtest/outputs_short_spot/baseline/cohorts").glob("*/checkpoint.json"))
    for checkpoint in checkpoints:
        record = json.loads(checkpoint.read_text(encoding="utf-8"))
        recorded_raw = record.get("signature", {}).get("inputs_and_code", {})
        recorded = {str(key).replace("\\", "/"): value for key, value in recorded_raw.items()}
        for name in source_names:
            expected = recorded.get(name)
            actual = current[name]
            if expected != actual:
                mismatches.append({
                    "cohort": checkpoint.parent.name,
                    "source": name,
                    "checkpoint_sha256": expected,
                    "current_sha256": actual,
                    "status": "missing_recorded_hash" if expected is None else "mismatch",
                })

    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Pre-fix immutable snapshot of historical evidence; hashes do not establish source equivalence.",
        "preserved_directories": list(preserved),
        "raw_input_sha256": input_hashes,
        "artifact_sha256": artifact_hashes,
        "current_source_sha256_at_audit": current,
        "checkpoint_source_comparisons": mismatches,
        "historical_exact_source_status": "unresolved" if mismatches else "matched_current_source",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    return audit
