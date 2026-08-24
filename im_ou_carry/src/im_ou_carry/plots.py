"""Diagnostic chart generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_curve(fits: pd.DataFrame, date: object, path: Path, title_prefix: str = "") -> None:
    curve = fits.loc[fits["date"] == pd.Timestamp(date)].sort_values("sessions_to_expiry")
    if curve.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["sessions_to_expiry"], 100 * curve["implied_carry"], "o-", label="Observed")
    ax.plot(curve["sessions_to_expiry"], 100 * curve["fitted_carry"], "s--", label="OU fitted")
    ax.set(xlabel="Trading sessions to expiry", ylabel="Annualized carry (%)", title=f"{title_prefix}{pd.Timestamp(date).date()}")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_selected_curves(fits: pd.DataFrame, dates: Iterable[object], path: Path, title: str) -> None:
    dates = list(dates)
    if not dates:
        return
    fig, axes = plt.subplots(len(dates), 1, figsize=(8, max(4, 3.2 * len(dates))), squeeze=False)
    for ax, date in zip(axes[:, 0], dates, strict=True):
        curve = fits.loc[fits["date"] == pd.Timestamp(date)].sort_values("sessions_to_expiry")
        ax.plot(curve["sessions_to_expiry"], 1e4 * curve["implied_carry"], "o-", label="Observed")
        ax.plot(curve["sessions_to_expiry"], 1e4 * curve["fitted_carry"], "s--", label="OU fitted")
        ax.set_title(str(pd.Timestamp(date).date()))
        ax.set_ylabel("Carry (bp)")
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Trading sessions to expiry")
    axes[0, 0].legend()
    fig.suptitle(title)
    _save(fig, path)


def plot_residuals(fits: pd.DataFrame, time_series: pd.DataFrame, maturity: pd.DataFrame, chart_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(time_series["date"], 1e4 * time_series["mean_carry_residual"], linewidth=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Daily mean carry residual", ylabel="Basis points", xlabel="Date")
    ax.grid(alpha=0.25)
    _save(fig, chart_dir / "residual_time_series.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(fits["sessions_to_expiry"], 1e4 * fits["carry_residual"], s=8, alpha=0.2)
    if not maturity.empty:
        positions = np.arange(len(maturity))
        ax2 = ax.twiny()
        ax2.plot(positions, 1e4 * maturity["mean_carry_residual"], "rD-", label="Bucket mean")
        ax2.set_xticks(positions, maturity["maturity_bucket"].astype(str), rotation=30)
        ax2.set_xlabel("Maturity-bucket mean")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Carry residuals by maturity", xlabel="Trading sessions to expiry", ylabel="Basis points")
    ax.grid(alpha=0.25)
    _save(fig, chart_dir / "residuals_by_maturity.png")


def plot_acf(acf_table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(acf_table["lag"], acf_table["acf"], width=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="ACF of daily mean carry residual", xlabel="Lag", ylabel="ACF")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_states(states: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(states["date"], 100 * states["filtered_state"], label="Filtered carry state")
    lower = 100 * (states["filtered_state"] - 1.96 * states["filtered_std"])
    upper = 100 * (states["filtered_state"] + 1.96 * states["filtered_std"])
    ax.fill_between(states["date"], lower, upper, alpha=0.2, label="95% interval")
    ax.set(title="Filtered instantaneous carry state", xlabel="Date", ylabel="Annualized carry (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_rolling_parameters(rolling: pd.DataFrame, path: Path) -> None:
    if rolling.empty:
        return
    columns = ["kappa", "theta", "eta", "sigma_epsilon"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, column in zip(axes.ravel(), columns, strict=True):
        ax.plot(rolling["window_end"], rolling[column], marker="o", markersize=3)
        ax.set_title(column)
        ax.grid(alpha=0.25)
    fig.suptitle("Rolling parameter stability")
    _save(fig, path)

