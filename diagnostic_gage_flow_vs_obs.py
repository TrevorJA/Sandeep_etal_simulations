"""Diagnostic: compare ensemble gage flows at Trenton & Montague with USGS observations."""
import os
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pywrdrb import Data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "pywrdrb", "inputs")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

GAGE_FILE = os.path.join(INPUT_DIR, "gage_flow_mgd.hdf5")
NODES = ["delMontague", "delTrenton"]
NODE_LABELS = {"delMontague": "Delaware at Montague", "delTrenton": "Delaware at Trenton"}


def load_ensemble_annual_means(node):
    """Load all realizations and compute annual mean flow for each."""
    with h5py.File(GAGE_FILE, "r") as f:
        real_ids = list(f[node].attrs["column_labels"])
        dates = pd.to_datetime(f[node]["date"][:].astype(str))
        all_annual = []
        for r in real_ids:
            s = pd.Series(f[node][r][:], index=dates)
            annual = s.resample("YS").mean()
            all_annual.append(annual)
    return pd.DataFrame({r: a for r, a in zip(real_ids, all_annual)})


def main():
    # Load observations
    d = Data(results_sets=["major_flow"])
    d.load_observations()
    obs = d.major_flow["obs"][0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for ax, node in zip(axes, NODES):
        label = NODE_LABELS[node]

        # Ensemble annual means
        ens_annual = load_ensemble_annual_means(node)
        years = ens_annual.index.year

        # Obs annual means (overlap period)
        obs_annual = obs[node].resample("YS").mean()
        obs_years = obs_annual.index.year

        # Ensemble envelope
        q05 = ens_annual.quantile(0.05, axis=1)
        q25 = ens_annual.quantile(0.25, axis=1)
        q50 = ens_annual.quantile(0.50, axis=1)
        q75 = ens_annual.quantile(0.75, axis=1)
        q95 = ens_annual.quantile(0.95, axis=1)

        ax.fill_between(years, q05, q95, alpha=0.15, color="C0", label="5-95th pctile")
        ax.fill_between(years, q25, q75, alpha=0.3, color="C0", label="25-75th pctile")
        ax.plot(years, q50, color="C0", lw=1.2, label="Ensemble median")

        # Observations
        mask = (obs_years >= years.min()) & (obs_years <= years.max())
        ax.plot(obs_years[mask], obs_annual.values[mask], color="k", lw=1.2,
                label="USGS observed")

        # Summary stats text
        overlap = (obs_years >= years.min()) & (obs_years <= years.max())
        obs_mean = obs_annual.values[overlap].mean()
        ens_mean = q50.mean()
        ratio = ens_mean / obs_mean
        ax.text(0.02, 0.95,
                f"Obs mean: {obs_mean:.0f} MGD\nEns median: {ens_mean:.0f} MGD\nRatio: {ratio:.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

        ax.set_ylabel("Annual mean flow (MGD)")
        ax.set_title(label)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(years.min(), years.max())

    axes[-1].set_xlabel("Year")
    fig.tight_layout()
    outpath = os.path.join(FIG_DIR, "gage_flow_vs_obs.png")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
