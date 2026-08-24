"""Headless diagnostic chart generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_latest_curves(fits: pd.DataFrame, path: Path) -> None:
    date = fits["date"].max()
    work = fits.loc[fits["date"] == date]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    observed = work.drop_duplicates(["date", "contract"]).sort_values("sessions_to_expiry")
    ax.plot(observed["sessions_to_expiry"], 1e4 * observed["implied_carry"], "ko-", label="Observed")
    for model, group in work.groupby("model", sort=False):
        group = group.sort_values("sessions_to_expiry")
        ax.plot(group["sessions_to_expiry"], 1e4 * group["fitted_carry"], "o--", label=model)
    ax.set(title=f"Latest carry curve: {pd.Timestamp(date).date()}", xlabel="Trading sessions to expiry", ylabel="Carry (bp)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    _save(fig, path)


def plot_selected_curves(fits: pd.DataFrame, dates: list[object], path: Path, title: str) -> None:
    if not dates:
        return
    fig, axes = plt.subplots(len(dates), 1, figsize=(9, 3.3 * len(dates)), squeeze=False)
    for ax, date in zip(axes[:, 0], dates, strict=True):
        work = fits.loc[fits["date"] == pd.Timestamp(date)]
        observed = work.drop_duplicates(["date", "contract"]).sort_values("sessions_to_expiry")
        ax.plot(observed["sessions_to_expiry"], 1e4 * observed["implied_carry"], "ko-", label="Observed")
        for model, group in work.groupby("model", sort=False):
            group = group.sort_values("sessions_to_expiry")
            ax.plot(group["sessions_to_expiry"], 1e4 * group["fitted_carry"], "--", label=model)
        ax.set(title=str(pd.Timestamp(date).date()), ylabel="Carry (bp)"); ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Trading sessions to expiry")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(title)
    _save(fig, path)


def plot_latest_futures(fits: pd.DataFrame, path: Path) -> None:
    date = fits["date"].max()
    work = fits.loc[fits["date"] == date]
    observed = work.drop_duplicates(["date", "contract"]).sort_values("sessions_to_expiry")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(observed["sessions_to_expiry"], observed["futures_price"], "ko-", label="Observed")
    for model, group in work.groupby("model", sort=False):
        group = group.sort_values("sessions_to_expiry")
        ax.plot(group["sessions_to_expiry"], group["fitted_futures_price"], "o--", label=model)
    ax.set(title=f"Latest fitted futures prices: {pd.Timestamp(date).date()}", xlabel="Trading sessions to expiry", ylabel="Index points")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    _save(fig, path)


def plot_states(states: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(len(states["model"].unique()), 1, figsize=(10, 3.5 * len(states["model"].unique())), sharex=True, squeeze=False)
    for ax, (model, group) in zip(axes[:, 0], states.groupby("model", sort=False), strict=True):
        ax.plot(group["date"], 100 * group["filtered_state"], label="Filtered")
        if "smoothed_state" in group:
            ax.plot(group["date"], 100 * group["smoothed_state"], alpha=0.7, label="Smoothed (historical only)")
        ax.fill_between(group["date"], 100 * (group["filtered_state"] - 1.96 * group["filtered_std"]), 100 * (group["filtered_state"] + 1.96 * group["filtered_std"]), alpha=0.15)
        ax.set(title=model, ylabel="Carry state (%)"); ax.grid(alpha=0.25); ax.legend()
    axes[-1, 0].set_xlabel("Date")
    _save(fig, path)


def plot_profiles(profiles: pd.DataFrame, path: Path) -> None:
    grouped = list(profiles.groupby("mode", sort=False))
    fig, axes = plt.subplots(1, len(grouped), figsize=(6 * len(grouped), 4.8), squeeze=False)
    for ax, (mode, group) in zip(axes[0], grouped, strict=True):
        ax.plot(group["rho"], group["lr_statistic"], "o-", markersize=4, label=mode)
        ax.axhline(3.841459, color="black", linestyle="--", label="95% LR cutoff")
        upper = max(4.5, min(30.0, 1.05 * float(group["lr_statistic"].max())))
        ax.set_ylim(bottom=-0.05 * upper, top=upper)
        ax.set(title=f"{mode.capitalize()} likelihood", xlabel="rho", ylabel="2(logL max - logL rho)")
        ax.grid(alpha=0.25); ax.legend()
    fig.suptitle("Profile likelihood for rho (tails clipped in joint panel)")
    _save(fig, path)


def plot_innovations(innovations: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(innovations["standardized_return_residual"], innovations["standardized_curve_innovation"], s=10, alpha=0.35)
    axes[0].set(xlabel="Standardized stock-return residual", ylabel="Standardized filtered curve innovation", title="Innovation co-movement")
    axes[0].grid(alpha=0.25)
    axes[1].plot(innovations["date"], innovations["rolling_correlation"])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Date", ylabel="Correlation", title="Rolling innovation correlation")
    axes[1].grid(alpha=0.25)
    _save(fig, path)


def plot_residuals(time_series: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for model, group in time_series.groupby("model", sort=False):
        ax.plot(group["date"], 1e4 * group["mean_carry_residual"], linewidth=0.8, label=model)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Daily mean carry residual", xlabel="Date", ylabel="Basis points")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    _save(fig, path)


def plot_acf(acf_table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for model, group in acf_table.groupby("model", sort=False):
        ax.plot(group["lag"], group["acf"], "o-", markersize=3, label=model)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="ACF of daily mean carry residual", xlabel="Lag", ylabel="ACF")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    _save(fig, path)


def plot_rolling(rolling: pd.DataFrame, path: Path) -> None:
    if rolling.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ax, parameter in zip(axes.ravel(), ["kappa", "theta", "eta", "rho", "sigma_epsilon", "mu"], strict=True):
        for model, group in rolling.groupby("model", sort=False):
            if parameter in group:
                ax.plot(group["window_end"], group[parameter], "o-", markersize=3, label=model)
        ax.set_title(parameter); ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    _save(fig, path)
