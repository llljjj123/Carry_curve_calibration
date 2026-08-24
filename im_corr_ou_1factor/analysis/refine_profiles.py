"""Recompute rho profiles from saved full-sample optima without rerunning all diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from im_corr_ou_1factor.config import load_config  # noqa: E402
from im_corr_ou_1factor.data import acquire_data, normalized_market_panel  # noqa: E402
from im_corr_ou_1factor.estimation import estimate_model, profile_rho  # noqa: E402
from im_corr_ou_1factor.model import OUParams  # noqa: E402
from im_corr_ou_1factor.pipeline import gap_function  # noqa: E402
from im_corr_ou_1factor.plots import plot_profiles  # noqa: E402
from im_corr_ou_1factor.quality import prepare_implied_carry  # noqa: E402


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    output_dir = Path(config["project"]["output_dir"])
    chart_dir = Path(config["project"]["chart_dir"])
    summary_path = output_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    spot, futures, _ = acquire_data(config)
    panel, _ = prepare_implied_carry(normalized_market_panel(spot, futures, config), config)
    coarse_cfg = config["estimation"]["rho_profile_grid"]
    coarse = np.linspace(float(coarse_cfg["start"]), float(coarse_cfg["stop"]), int(coarse_cfg["points"]))
    refine_cfg = config["estimation"]["joint_rho_refinement_grid"]
    refine = np.linspace(float(refine_cfg["start"]), float(refine_cfg["stop"]), int(refine_cfg["points"]))
    frames = []
    summaries = []
    for name, mode, grid in (
        ("exact_corr_curve", "curve", coarse),
        ("exact_corr_joint", "joint", np.unique(np.concatenate([coarse, refine]))),
    ):
        saved = OUParams(**summary["models"][name]["parameters"])
        unrestricted = estimate_model(
            panel,
            name=name,
            mode=mode,
            variant="exact",
            sigma=float(config["model"]["stock_volatility"]),
            gap_function=gap_function(config),
            free_rho=True,
            starts=1,
            maxiter=int(config["estimation"]["optimizer_maxiter"]),
            compute_standard_errors=False,
            supplied_starts=[saved],
        )
        profile, profile_summary = profile_rho(
            panel,
            unrestricted,
            grid,
            gap_function=gap_function(config),
            starts=1,
            maxiter=int(config["estimation"]["optimizer_maxiter"]),
            seed=int(config["estimation"]["random_seed"]),
        )
        profile["model"] = name
        profile["mode"] = mode
        frames.append(profile)
        summaries.append({"model": name, "mode": mode, **profile_summary})
    profiles = pd.concat(frames, ignore_index=True)
    profile_summaries = pd.DataFrame(summaries)
    profiles.to_csv(output_dir / "rho_profile_likelihood.csv", index=False, encoding="utf-8-sig")
    profile_summaries.to_csv(output_dir / "rho_profile_confidence_intervals.csv", index=False, encoding="utf-8-sig")
    plot_profiles(profiles, chart_dir / "rho_profile_likelihood.png")
    summary["profile_confidence_intervals"] = profile_summaries.to_dict(orient="records")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(profile_summaries.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

