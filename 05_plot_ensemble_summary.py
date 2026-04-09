"""Plot ensemble summary: NYC aggregate storage and Montague streamflow.

Produces a 2-panel figure:
  Top:    NYC aggregate reservoir storage (MG) - ensemble range, median, and observed
  Bottom: Delaware at Montague streamflow (MGD, log scale) - ensemble range, median, and observed
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pywrdrb import Data

from config import OUTPUT_DIR, FIG_DIR, FLOW_TYPE

NYC_RESERVOIRS = ["cannonsville", "pepacton", "neversink"]

def main():
    output_file = os.path.join(OUTPUT_DIR, f"{FLOW_TYPE}.hdf5")
    assert os.path.exists(output_file), f"Output file not found: {output_file}"

    # Load simulation output and observations
    data = Data(
        output_filenames=[output_file],
        results_sets=["res_storage", "major_flow"],
        print_status=True,
    )
    data.load_output()
    data.load_observations()

    sim_storage = data.res_storage[FLOW_TYPE]
    sim_flow = data.major_flow[FLOW_TYPE]
    obs_storage = data.res_storage["obs"][0]
    obs_flow = data.major_flow["obs"][0]

    n_scenarios = len(sim_storage)
    dates = sim_storage[0].index

    # Build ensemble arrays: NYC aggregate storage
    nyc_storage_ensemble = np.column_stack([
        sim_storage[s][NYC_RESERVOIRS].sum(axis=1).values
        for s in range(n_scenarios)
    ])

    # Build ensemble arrays: Montague flow
    montague_ensemble = np.column_stack([
        sim_flow[s]["delMontague"].values
        for s in range(n_scenarios)
    ])

    # Compute statistics
    nyc_median = np.median(nyc_storage_ensemble, axis=1)
    nyc_q10 = np.percentile(nyc_storage_ensemble, 10, axis=1)
    nyc_q90 = np.percentile(nyc_storage_ensemble, 90, axis=1)
    nyc_min = nyc_storage_ensemble.min(axis=1)
    nyc_max = nyc_storage_ensemble.max(axis=1)

    mont_median = np.median(montague_ensemble, axis=1)
    mont_q10 = np.percentile(montague_ensemble, 10, axis=1)
    mont_q90 = np.percentile(montague_ensemble, 90, axis=1)
    mont_min = montague_ensemble.min(axis=1)
    mont_max = montague_ensemble.max(axis=1)

    # Observed NYC aggregate storage
    obs_nyc_cols = [c for c in NYC_RESERVOIRS if c in obs_storage.columns]
    obs_nyc = obs_storage[obs_nyc_cols].sum(axis=1)

    # Observed Montague flow
    obs_mont = obs_flow["delMontague"]

    # Trim observations to simulation period for plotting
    sim_start, sim_end = dates[0], dates[-1]

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top panel: NYC aggregate storage
    ax1.fill_between(dates, nyc_min, nyc_max, alpha=0.15, color="steelblue", label="Ensemble range")
    ax1.fill_between(dates, nyc_q10, nyc_q90, alpha=0.3, color="steelblue", label="10th-90th pctl")
    ax1.plot(dates, nyc_median, color="steelblue", lw=0.8, label="Ensemble median")
    ax1.plot(obs_nyc.index, obs_nyc.values, color="black", lw=0.6, alpha=0.8, label="Observed")
    ax1.set_ylabel("NYC Aggregate Storage (MG)")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_title(f"Ensemble Summary ({FLOW_TYPE}, {n_scenarios} members)")

    # Bottom panel: Montague streamflow (log scale)
    ax2.fill_between(dates, mont_min, mont_max, alpha=0.15, color="steelblue", label="Ensemble range")
    ax2.fill_between(dates, mont_q10, mont_q90, alpha=0.3, color="steelblue", label="10th-90th pctl")
    ax2.plot(dates, mont_median, color="steelblue", lw=0.5, label="Ensemble median")
    ax2.plot(obs_mont.index, obs_mont.values, color="black", lw=0.4, alpha=0.8, label="Observed")
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=500)  # Set a reasonable lower limit for log scale
    ax2.set_ylabel("Montague Streamflow (MGD)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right", fontsize=8)

    ax1.set_xlim(sim_start, sim_end)

    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    outpath = os.path.join(FIG_DIR, "ensemble_summary.png")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
