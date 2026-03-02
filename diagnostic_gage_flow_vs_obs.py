"""Diagnostic: compare ensemble GAGE FLOW with USGS observations.

All plots show cumulative gage flow (MGD) — NOT catchment inflow.
Focuses on delLordville, delMontague, delTrenton.

Produces figures in figures/<flow_type>/:
  1. Annual mean gage flow vs USGS observations
  2. Daily gage flow snapshots for specific 2-year periods
  3. Mean daily gage flow by calendar month
"""
import os
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pywrdrb import Data

from config import GAGE_FLOW_FILE, FIG_DIR, FLOW_TYPE

NODES = ["delLordville", "delMontague", "delTrenton"]
NODE_LABELS = {
    "delLordville": "Delaware at Lordville",
    "delMontague": "Delaware at Montague",
    "delTrenton": "Delaware at Trenton",
}

# 2-year snapshot windows (start_year, end_year)
# Pick periods where observations overlap with ensemble data (1945-2018)
SNAPSHOT_PERIODS = [
    (1965, 1966),   # includes 1960s drought
    (2005, 2006),   # recent, before delLordville obs starts
    (2010, 2011),   # all 3 nodes have obs
]


def load_ensemble_daily(node):
    """Load all realizations of gage flow as a DataFrame (dates x realizations)."""
    with h5py.File(GAGE_FLOW_FILE, "r") as f:
        real_ids = list(f[node].attrs["column_labels"])
        dates = pd.to_datetime(f[node]["date"][:].astype(str))
        data = {r: f[node][r][:] for r in real_ids}
    return pd.DataFrame(data, index=dates)


def plot_annual_means(obs, fname):
    """Annual mean gage flow: ensemble envelope vs USGS observations."""
    fig, axes = plt.subplots(len(NODES), 1, figsize=(12, 3.5 * len(NODES)), sharex=True)

    for ax, node in zip(axes, NODES):
        ens_daily = load_ensemble_daily(node)
        ens_annual = ens_daily.resample("YS").mean()
        years = ens_annual.index.year

        q05 = ens_annual.quantile(0.05, axis=1)
        q25 = ens_annual.quantile(0.25, axis=1)
        q50 = ens_annual.quantile(0.50, axis=1)
        q75 = ens_annual.quantile(0.75, axis=1)
        q95 = ens_annual.quantile(0.95, axis=1)

        ax.fill_between(years, q05, q95, alpha=0.15, color="C0", label="5-95th pctile")
        ax.fill_between(years, q25, q75, alpha=0.3, color="C0", label="25-75th pctile")
        ax.plot(years, q50, color="C0", lw=1.2, label="Ensemble median")

        if node in obs.columns:
            obs_annual = obs[node].resample("YS").mean().dropna()
            obs_years = obs_annual.index.year
            mask = (obs_years >= years.min()) & (obs_years <= years.max())
            ax.plot(obs_years[mask], obs_annual.values[mask], color="k", lw=1.2,
                    label="USGS observed")

            # Summary: ratio of ensemble median to obs over overlap
            overlap_years = obs_years[mask]
            if len(overlap_years) > 0:
                ens_overlap = q50.loc[q50.index.year.isin(overlap_years)]
                obs_overlap = obs_annual.loc[obs_annual.index.year.isin(overlap_years)]
                if len(ens_overlap) > 0 and len(obs_overlap) > 0:
                    ratio = ens_overlap.mean() / obs_overlap.mean()
                    ax.text(0.02, 0.95,
                            f"Ensemble/Obs ratio: {ratio:.2f}",
                            transform=ax.transAxes, va="top", fontsize=9,
                            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

        ax.set_ylabel("Annual mean gage flow (MGD)")
        ax.set_title(NODE_LABELS[node])
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(years.min(), years.max())

    axes[-1].set_xlabel("Year")
    fig.suptitle(f"Annual mean GAGE FLOW vs USGS observations — {FLOW_TYPE}",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_daily_snapshots(obs, fname):
    """Daily gage flow for specific 2-year periods: ensemble envelope vs obs."""
    n_periods = len(SNAPSHOT_PERIODS)
    n_nodes = len(NODES)

    fig, axes = plt.subplots(n_periods, n_nodes,
                             figsize=(5 * n_nodes, 3.5 * n_periods),
                             squeeze=False)

    for row, (y_start, y_end) in enumerate(SNAPSHOT_PERIODS):
        period_start = f"{y_start}-01-01"
        period_end = f"{y_end}-12-31"

        for col, node in enumerate(NODES):
            ax = axes[row, col]

            ens_daily = load_ensemble_daily(node)
            mask = (ens_daily.index >= period_start) & (ens_daily.index <= period_end)
            ens_sub = ens_daily.loc[mask]

            if len(ens_sub) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            dates = ens_sub.index
            q05 = ens_sub.quantile(0.05, axis=1)
            q25 = ens_sub.quantile(0.25, axis=1)
            q50 = ens_sub.quantile(0.50, axis=1)
            q75 = ens_sub.quantile(0.75, axis=1)
            q95 = ens_sub.quantile(0.95, axis=1)

            ax.fill_between(dates, q05, q95, alpha=0.15, color="C0")
            ax.fill_between(dates, q25, q75, alpha=0.3, color="C0")
            ax.plot(dates, q50, color="C0", lw=0.8, label="Ensemble median")

            if node in obs.columns:
                obs_sub = obs[node].loc[period_start:period_end].dropna()
                if len(obs_sub) > 0:
                    ax.plot(obs_sub.index, obs_sub.values, color="k", lw=0.8,
                            label="USGS observed")

            if row == 0:
                ax.set_title(NODE_LABELS[node], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{y_start}-{y_end}\nGage flow (MGD)", fontsize=9)
            if row == n_periods - 1:
                ax.tick_params(axis="x", rotation=30)
            else:
                ax.set_xticklabels([])

            if row == 0 and col == n_nodes - 1:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"Daily GAGE FLOW: ensemble vs observations — {FLOW_TYPE}",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_monthly_climatology(obs, fname):
    """Mean daily gage flow by calendar month: ensemble vs observations."""
    fig, axes = plt.subplots(1, len(NODES), figsize=(5 * len(NODES), 4.5),
                             sharey=False, squeeze=False)
    axes = axes[0]

    months = np.arange(1, 13)
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    for ax, node in zip(axes, NODES):
        ens_daily = load_ensemble_daily(node)

        # Mean daily flow per month per realization
        ens_monthly = ens_daily.groupby(ens_daily.index.month).mean()

        q25 = ens_monthly.quantile(0.25, axis=1)
        q50 = ens_monthly.quantile(0.50, axis=1)
        q75 = ens_monthly.quantile(0.75, axis=1)

        ax.fill_between(months, q25, q75, alpha=0.3, color="C0", label="25-75th pctile")
        ax.plot(months, q50, "o-", color="C0", lw=1.2, markersize=4, label="Ensemble median")

        if node in obs.columns:
            obs_monthly = obs[node].groupby(obs[node].index.month).mean()
            ax.plot(months, obs_monthly.values, "s-", color="k", lw=1.2,
                    markersize=4, label="USGS observed")

        ax.set_xlabel("Month")
        ax.set_title(NODE_LABELS[node], fontsize=10)
        ax.set_xticks(months)
        ax.set_xticklabels(month_labels, fontsize=9)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Mean daily gage flow (MGD)")
    fig.suptitle(f"Monthly mean GAGE FLOW — {FLOW_TYPE}", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    print(f"Dataset: {FLOW_TYPE}")
    print(f"Figures → {FIG_DIR}\n")

    d = Data(results_sets=["major_flow"])
    d.load_observations()
    obs = d.major_flow["obs"][0]

    print("1. Annual mean gage flow vs observations...")
    plot_annual_means(obs, "gage_flow_annual_vs_obs.png")

    print("2. Daily gage flow snapshots (2-year periods)...")
    plot_daily_snapshots(obs, "gage_flow_daily_snapshots.png")

    print("3. Monthly mean gage flow by calendar month...")
    plot_monthly_climatology(obs, "gage_flow_monthly_mean.png")


if __name__ == "__main__":
    main()
