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


def plot_curve_comparison(
    observed: pd.DataFrame,
    one_factor: pd.DataFrame,
    two_factor: pd.DataFrame,
    date: object,
    path: Path,
    title_prefix: str = "",
) -> None:
    date = pd.Timestamp(date)
    raw = observed.loc[observed["date"] == date].sort_values("sessions_to_expiry")
    one = one_factor.loc[one_factor["date"] == date].sort_values("sessions_to_expiry")
    two = two_factor.loc[two_factor["date"] == date].sort_values("sessions_to_expiry")
    if raw.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(raw["sessions_to_expiry"], 100 * raw["implied_carry"], "ko-", label="Observed")
    ax.plot(one["sessions_to_expiry"], 100 * one["fitted_carry"], "s--", label="One-factor")
    ax.plot(two["sessions_to_expiry"], 100 * two["fitted_carry"], "D-.", label="Two-factor")
    ax.set(
        xlabel="Trading sessions to expiry",
        ylabel="Annualized carry (%)",
        title=f"{title_prefix}{date.date()}",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_factor_states(states: pd.DataFrame, theta: float, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(states["date"], 100 * states["filtered_slow_state"], label="Slow factor")
    axes[0].plot(states["date"], 100 * states["filtered_fast_state"], label="Fast factor", alpha=0.8)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Factor state (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    carry = states["filtered_instantaneous_carry"]
    std = states["filtered_instantaneous_std"]
    axes[1].plot(states["date"], 100 * carry, label="Instantaneous carry")
    axes[1].fill_between(states["date"], 100 * (carry - 1.96 * std), 100 * (carry + 1.96 * std), alpha=0.2)
    axes[1].axhline(100 * theta, color="black", linestyle="--", linewidth=0.8, label="theta")
    axes[1].set(xlabel="Date", ylabel="Carry (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("Filtered slow/fast factors and instantaneous carry")
    _save(fig, path)


def plot_factor_loadings(kappa_slow: float, kappa_fast: float, path: Path) -> None:
    from .kalman import maturity_loading

    sessions = np.arange(1, 366)
    tau = sessions / 244.0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sessions, maturity_loading(kappa_slow, tau), label=f"Slow loading (kappa={kappa_slow:.2f})")
    ax.plot(sessions, maturity_loading(kappa_fast, tau), label=f"Fast loading (kappa={kappa_fast:.2f})")
    ax.set(title="Two-factor maturity loadings", xlabel="Trading sessions to expiry", ylabel="Loading")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_two_factor_rolling(rolling: pd.DataFrame, path: Path) -> None:
    if rolling.empty:
        return
    columns = ["kappa_slow", "kappa_fast", "theta", "eta_slow", "eta_fast", "sigma_epsilon"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    for ax, column in zip(axes.ravel(), columns, strict=True):
        ax.plot(rolling["window_end"], rolling[column], marker="o", markersize=3)
        ax.set_title(column)
        ax.grid(alpha=0.25)
    fig.suptitle("Rolling two-factor parameter stability")
    _save(fig, path)
