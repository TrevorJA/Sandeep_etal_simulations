"""Plot ensemble summary: NYC aggregate storage and Montague/Trenton streamflow.

Produces 3x2 figures (timeseries on left, CDFs on right) for the full simulation
period and zoomed windows starting in 2000 and 2010. Each row covers:
  Top:    NYC aggregate reservoir storage (MG)
  Middle: Delaware at Montague streamflow (MGD, log scale)
  Bottom: Delaware at Trenton streamflow (MGD, log scale)

Each CDF is computed from the same date window as its companion timeseries panel,
so zoomed plots show distributions for only that period.
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
SMOOTHING_WINDOW = 7  # days


def smooth_array(arr, window):
    """Apply a centered rolling mean along axis 0 (time) of a 1- or 2-D array."""
    df = pd.DataFrame(arr)
    return df.rolling(window=window, min_periods=1, center=True).mean().values


def smooth_series(s, window):
    return s.rolling(window=window, min_periods=1, center=True).mean()


def slice_window(dates, ensemble, obs, xlim):
    dates = pd.DatetimeIndex(dates)
    mask = (dates >= xlim[0]) & (dates <= xlim[1])
    obs_mask = (obs.index >= xlim[0]) & (obs.index <= xlim[1])
    return dates[mask], ensemble[mask], obs.loc[obs_mask]


def plot_row(ax_ts, ax_cdf, dates, ensemble, obs, label,
             ylim=None, log=False, title=None):
    median = np.median(ensemble, axis=1)
    q10 = np.percentile(ensemble, 10, axis=1)
    q90 = np.percentile(ensemble, 90, axis=1)
    mn = ensemble.min(axis=1)
    mx = ensemble.max(axis=1)

    ts_lw = 0.5 if log else 0.8
    obs_lw = 0.4 if log else 0.6

    # --- Timeseries ---
    ax_ts.fill_between(dates, mn, mx, alpha=0.15, color="steelblue",
                       label="Ensemble range")
    ax_ts.fill_between(dates, q10, q90, alpha=0.3, color="steelblue",
                       label="10th-90th pctl")
    ax_ts.plot(dates, median, color="steelblue", lw=ts_lw,
               label="Ensemble median")
    ax_ts.plot(obs.index, obs.values, color="black", lw=obs_lw, alpha=0.8,
               label="Observed")
    if log:
        ax_ts.set_yscale("log")
    if ylim is not None:
        ax_ts.set_ylim(*ylim)
    ax_ts.set_ylabel(label)
    if title:
        ax_ts.set_title(title)

    # --- CDF (x = cumulative probability, y = value) ---
    # Sort each member's series independently to get its empirical CDF, then
    # take min/max/percentiles across members at each probability level.
    sorted_ens = np.sort(ensemble, axis=0)
    n_time = sorted_ens.shape[0]
    p = np.arange(1, n_time + 1) / n_time
    cdf_min = sorted_ens.min(axis=1)
    cdf_max = sorted_ens.max(axis=1)
    cdf_q10 = np.percentile(sorted_ens, 10, axis=1)
    cdf_q90 = np.percentile(sorted_ens, 90, axis=1)
    cdf_median = np.median(sorted_ens, axis=1)

    ax_cdf.fill_between(p, cdf_min, cdf_max, alpha=0.15, color="steelblue")
    ax_cdf.fill_between(p, cdf_q10, cdf_q90, alpha=0.3, color="steelblue")
    ax_cdf.plot(p, cdf_median, color="steelblue", lw=0.9)

    obs_clean = obs.dropna().values
    if len(obs_clean) > 0:
        obs_sorted = np.sort(obs_clean)
        obs_p = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted)
        ax_cdf.plot(obs_p, obs_sorted, color="black", lw=0.9, alpha=0.9)

    if log:
        ax_cdf.set_yscale("log")
    if ylim is not None:
        ax_cdf.set_ylim(*ylim)
    ax_cdf.set_xlim(0, 1)
    ax_cdf.tick_params(labelleft=False)
    ax_cdf.grid(True, alpha=0.3)


def plot_ensemble(dates, ens_nyc, ens_mont, ens_trenton,
                  obs_nyc, obs_mont, obs_trenton,
                  xlim, n_scenarios, outpath, title_suffix):
    dates_n, ens_nyc_w, obs_nyc_w = slice_window(dates, ens_nyc, obs_nyc, xlim)
    dates_m, ens_mont_w, obs_mont_w = slice_window(dates, ens_mont, obs_mont, xlim)
    dates_t, ens_tre_w, obs_tre_w = slice_window(dates, ens_trenton, obs_trenton, xlim)

    # Recompute log-scale flow y-limits from the windowed data so the zoom view
    # is well-scaled for the visible period.
    mont_ylim = (np.percentile(ens_mont_w, 1), np.percentile(ens_mont_w, 99))
    tre_ylim = (np.percentile(ens_tre_w, 1), np.percentile(ens_tre_w, 99))

    fig, axes = plt.subplots(3, 2, figsize=(18, 11),
                             gridspec_kw={"width_ratios": [2.5, 1]})

    title = (f"Ensemble Summary ({FLOW_TYPE}, {n_scenarios} members, "
             f"{SMOOTHING_WINDOW}-day smoothing) - {title_suffix}")

    plot_row(axes[0, 0], axes[0, 1], dates_n, ens_nyc_w, obs_nyc_w,
             label="NYC Aggregate Storage (MG)", title=title)
    plot_row(axes[1, 0], axes[1, 1], dates_m, ens_mont_w, obs_mont_w,
             label="Montague Flow (MGD)", ylim=mont_ylim, log=True)
    plot_row(axes[2, 0], axes[2, 1], dates_t, ens_tre_w, obs_tre_w,
             label="Trenton Flow (MGD)", ylim=tre_ylim, log=True)

    axes[2, 0].set_xlabel("Date")
    axes[2, 1].set_xlabel("Cumulative probability")
    for ax in axes[:, 0]:
        ax.set_xlim(*xlim)
    for ax in axes[:2, 0]:
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in axes[:2, 1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.0), fontsize=10, frameon=True)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved {outpath}")


def main():
    output_file = os.path.join(OUTPUT_DIR, f"{FLOW_TYPE}.hdf5")
    assert os.path.exists(output_file), f"Output file not found: {output_file}"

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

    # Ensemble arrays (rows = time, cols = members)
    nyc_storage_ensemble = np.column_stack([
        sim_storage[s][NYC_RESERVOIRS].sum(axis=1).values
        for s in range(n_scenarios)
    ])
    montague_ensemble = np.column_stack([
        sim_flow[s]["delMontague"].values for s in range(n_scenarios)
    ])
    trenton_ensemble = np.column_stack([
        sim_flow[s]["delTrenton"].values for s in range(n_scenarios)
    ])

    # Smooth ensemble members along the time axis
    nyc_storage_ensemble = smooth_array(nyc_storage_ensemble, SMOOTHING_WINDOW)
    montague_ensemble = smooth_array(montague_ensemble, SMOOTHING_WINDOW)
    trenton_ensemble = smooth_array(trenton_ensemble, SMOOTHING_WINDOW)

    # Observations (smoothed)
    obs_nyc_cols = [c for c in NYC_RESERVOIRS if c in obs_storage.columns]
    obs_nyc = smooth_series(obs_storage[obs_nyc_cols].sum(axis=1), SMOOTHING_WINDOW)
    obs_mont = smooth_series(obs_flow["delMontague"], SMOOTHING_WINDOW)
    obs_trenton = smooth_series(obs_flow["delTrenton"], SMOOTHING_WINDOW)

    sim_start, sim_end = dates[0], dates[-1]
    os.makedirs(FIG_DIR, exist_ok=True)

    windows = [
        (pd.Timestamp(sim_start), pd.Timestamp(sim_end),
         "ensemble_summary_full.png",
         f"full period {pd.Timestamp(sim_start).year}-{pd.Timestamp(sim_end).year}"),
    ]
    for start_year, label in [(2000, "2000-end"), (2010, "2010-end")]:
        zoom_start = pd.Timestamp(f"{start_year}-01-01")
        if zoom_start >= pd.Timestamp(sim_end):
            print(f"Skipping {label} plot: simulation ends before {start_year} ({sim_end})")
            continue
        zoom_start = max(zoom_start, pd.Timestamp(sim_start))
        windows.append(
            (zoom_start, pd.Timestamp(sim_end),
             f"ensemble_summary_{label}.png",
             f"{zoom_start.year}-{pd.Timestamp(sim_end).year}")
        )

    for start, end, fname, suffix in windows:
        plot_ensemble(
            dates,
            nyc_storage_ensemble, montague_ensemble, trenton_ensemble,
            obs_nyc, obs_mont, obs_trenton,
            xlim=(start, end),
            n_scenarios=n_scenarios,
            outpath=os.path.join(FIG_DIR, fname),
            title_suffix=suffix,
        )


if __name__ == "__main__":
    main()
