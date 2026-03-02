"""Diagnostic: where and when do near-zero values appear in the ensemble data?

Focuses on major mainstem nodes (delLordville, delMontague, delTrenton).
Produces figures in figures/<flow_type>/:
  1. Side-by-side heatmaps: gage flow vs catchment inflow (% near-zero)
  2. Seasonal pattern of zeros in catchment inflow
  3. Console summary statistics
"""
import os
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CSV_DIR, INPUT_DIR, N_REALIZATIONS, FIG_DIR, FLOW_TYPE

ZERO_THRESH = 0.01

# Focus nodes
FOCUS_NODES = ["delLordville", "delMontague", "delTrenton"]
FOCUS_LABELS = {
    "delLordville": "Delaware at Lordville",
    "delMontague": "Delaware at Montague",
    "delTrenton": "Delaware at Trenton",
}

# All regression nodes for the summary
ALL_REGRESSION_NODES = [
    "delLordville", "delMontague", "delDRCanal", "delTrenton",
    "01425000", "01417000", "01436000",
    "wallenpaupack", "prompton", "shoholaMarsh", "mongaupeCombined",
    "01433500", "beltzvilleCombined", "01447800",
    "fewalter", "01449800", "hopatcong", "merrillCreek", "nockamixon",
]

MAJOR_NODES = {"delLordville", "delMontague", "delDRCanal", "delTrenton"}


def pct_near_zero(arr, thresh=ZERO_THRESH):
    return np.sum(np.abs(arr) <= thresh) / len(arr) * 100


def plot_heatmap_comparison(gage_file, catch_file, fname):
    """Side-by-side heatmaps: gage flow vs catchment inflow for focus nodes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.5), sharey=True)

    with h5py.File(gage_file, "r") as fg, h5py.File(catch_file, "r") as fc:
        real_ids = list(fg[FOCUS_NODES[0]].attrs["column_labels"])
        n_real = len(real_ids)

        for ax, fh, label in [(ax1, fg, "Gage flow (MGD)"),
                               (ax2, fc, "Catchment inflow (MGD)")]:
            matrix = np.zeros((len(FOCUS_NODES), n_real))
            for i, node in enumerate(FOCUS_NODES):
                for j, r in enumerate(real_ids):
                    matrix[i, j] = pct_near_zero(fh[node][r][:])

            im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
            ax.set_yticks(range(len(FOCUS_NODES)))
            ax.set_yticklabels([FOCUS_LABELS[n] for n in FOCUS_NODES], fontsize=9)
            ax.set_xlabel("Realization index")
            ax.set_title(label, fontsize=11)

            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if matrix[i, j] >= 99.5:
                        ax.plot(j, i, "x", color="black", markersize=3, markeredgewidth=0.5)

    cbar = fig.colorbar(im, ax=[ax1, ax2], shrink=0.7, pad=0.03)
    cbar.set_label("% days ≤ 0.01 MGD")
    fig.suptitle(f"Fraction of near-zero days per realization — {FLOW_TYPE}", fontsize=12)
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_seasonal_zeros(catch_file, fname):
    """Monthly breakdown of zero frequency in catchment inflow for focus nodes."""
    fig, axes = plt.subplots(1, len(FOCUS_NODES), figsize=(5 * len(FOCUS_NODES), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    with h5py.File(catch_file, "r") as f:
        dates = pd.to_datetime(f[FOCUS_NODES[0]]["date"][:].astype(str))
        months = dates.month
        real_ids = list(f[FOCUS_NODES[0]].attrs["column_labels"])

        for ax, node in zip(axes, FOCUS_NODES):
            month_data = {m: [] for m in range(1, 13)}
            for r in real_ids:
                vals = f[node][r][:]
                is_zero = np.abs(vals) <= ZERO_THRESH
                for m in range(1, 13):
                    mask = months == m
                    pct = np.mean(is_zero[mask]) * 100
                    month_data[m].append(pct)

            bp = ax.boxplot([month_data[m] for m in range(1, 13)],
                            positions=range(1, 13), widths=0.6,
                            patch_artist=True, showfliers=True,
                            flierprops=dict(marker=".", markersize=3))
            for patch in bp["boxes"]:
                patch.set_facecolor("lightcoral")
                patch.set_alpha(0.7)

            ax.set_xlabel("Month")
            ax.set_title(FOCUS_LABELS[node], fontsize=10)
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(["J", "F", "M", "A", "M", "J",
                                "J", "A", "S", "O", "N", "D"], fontsize=9)

    axes[0].set_ylabel("% days near-zero (per realization)")
    fig.suptitle(f"Seasonal pattern of zeros in catchment inflow — {FLOW_TYPE}",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def print_summary(catch_file):
    """Print summary statistics."""
    print(f"\n=== Zero-value summary ({FLOW_TYPE}) ===\n")

    with h5py.File(catch_file, "r") as f:
        real_ids = list(f[ALL_REGRESSION_NODES[0]].attrs["column_labels"])

        # Per-node stats
        catch_pcts = {}
        for node in ALL_REGRESSION_NODES:
            pcts = np.array([pct_near_zero(f[node][r][:]) for r in real_ids])
            catch_pcts[node] = pcts

    print("--- Catchment inflow: major mainstem nodes ---")
    for node in ALL_REGRESSION_NODES:
        if node not in MAJOR_NODES:
            continue
        pcts = catch_pcts[node]
        print(f"  {node:25s}: mean {pcts.mean():5.1f}%, max {pcts.max():5.1f}%, "
              f"{np.sum(pcts > 90):3d} realizations >90% zero")

    print("\n--- Catchment inflow: minor nodes with issues (>10% mean zero) ---")
    for node in ALL_REGRESSION_NODES:
        if node in MAJOR_NODES:
            continue
        pcts = catch_pcts[node]
        if pcts.mean() > 10:
            print(f"  {node:25s}: mean {pcts.mean():5.1f}%, max {pcts.max():5.1f}%, "
                  f"{np.sum(pcts > 90):3d} realizations >90% zero")

    # Crash analysis
    print(f"\n--- Realizations that would crash _fit_regression (>95% zero) ---")
    print(f"  (excluding delTrenton, which is zero by design)")
    fail_reals_major = set()
    fail_reals_minor = set()
    fail_by_node = {}
    n_real = len(real_ids)

    for node in ALL_REGRESSION_NODES:
        if node == "delTrenton":
            continue
        pcts = catch_pcts[node]
        fails = [j for j in range(n_real) if pcts[j] > 95]
        if fails:
            fail_by_node[node] = len(fails)
            for j in fails:
                if node in MAJOR_NODES:
                    fail_reals_major.add(j)
                else:
                    fail_reals_minor.add(j)

    all_fails = fail_reals_major | fail_reals_minor
    print(f"  Total: {len(all_fails)} / {n_real} realizations would crash")
    print(f"    Due to major nodes: {len(fail_reals_major)}")
    print(f"    Due to minor nodes only: {len(fail_reals_minor - fail_reals_major)}")
    if fail_by_node:
        print("  Breakdown:")
        for node, count in sorted(fail_by_node.items(), key=lambda x: -x[1]):
            tag = " *" if node in MAJOR_NODES else ""
            print(f"    {node:25s}: {count:3d} realizations{tag}")


def main():
    gage_file = os.path.join(INPUT_DIR, "gage_flow_mgd.hdf5")
    catch_file = os.path.join(INPUT_DIR, "catchment_inflow_mgd.hdf5")

    print(f"Dataset: {FLOW_TYPE}")
    print(f"Figures → {FIG_DIR}\n")

    print("1. Heatmap comparison: gage flow vs catchment inflow...")
    plot_heatmap_comparison(gage_file, catch_file, "zero_pct_heatmap.png")

    print("2. Seasonal pattern of zeros in catchment inflow...")
    plot_seasonal_zeros(catch_file, "zero_seasonal_catchment_inflow.png")

    print_summary(catch_file)


if __name__ == "__main__":
    main()
