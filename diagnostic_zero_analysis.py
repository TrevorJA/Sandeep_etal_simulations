"""Diagnostic plots: fraction of near-zero values across realizations and nodes.

Produces three figures in figures/:
  1. Heatmap of % near-zero in raw CSVs (mm/day, before unit conversion)
  2. Heatmap of % near-zero in gage_flow_mgd.hdf5
  3. Heatmap of % near-zero in catchment_inflow_mgd.hdf5
"""
import os
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "drb_streamflow_ensembles", "hybrid_finetuned")
INPUT_DIR = os.path.join(BASE_DIR, "pywrdrb", "inputs")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ZERO_THRESH = 0.01
N_REALIZATIONS = 160

# Regression nodes (used in predicted inflow regressions)
REGRESSION_NODES = [
    "01425000", "01417000", "delLordville", "01436000",
    "wallenpaupack", "prompton", "shoholaMarsh", "mongaupeCombined",
    "01433500", "delMontague", "beltzvilleCombined", "01447800",
    "fewalter", "01449800", "hopatcong", "merrillCreek", "nockamixon",
    "delDRCanal", 'delTrenton'
]


def pct_near_zero(arr, thresh=ZERO_THRESH):
    return np.sum(np.abs(arr) <= thresh) / len(arr) * 100


def make_heatmap(matrix, nodes, title, fname, realization_labels=None):
    """Heatmap: nodes (y) x realizations (x), colored by % near-zero."""
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(nodes, fontsize=8)
    ax.set_xlabel("Realization index")
    ax.set_ylabel("Node")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("% values ≤ 0.01")

    # Mark realizations that are 100% zero
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] >= 99.5:
                ax.plot(j, i, "x", color="black", markersize=3, markeredgewidth=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    # ----------------------------------------------------------------
    # 1. Raw CSVs (only regression nodes that have CSVs)
    # ----------------------------------------------------------------
    print("1. Analyzing raw CSVs...")
    csv_nodes = []
    csv_matrix_rows = []
    for node in REGRESSION_NODES:
        csv_path = os.path.join(CSV_DIR, f"pred_{node}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, index_col="date")
        sim_cols = [f"qsim_{i}" for i in range(1, N_REALIZATIONS + 1)]
        row = [pct_near_zero(df[col].values) for col in sim_cols]
        csv_nodes.append(node)
        csv_matrix_rows.append(row)

    csv_matrix = np.array(csv_matrix_rows)
    make_heatmap(csv_matrix, csv_nodes,
                 "Raw CSV: % values ≤ 0.01 mm/day (before conversion)",
                 "zero_pct_raw_csv.png")

    # ----------------------------------------------------------------
    # 2. Gage flow HDF5
    # ----------------------------------------------------------------
    print("2. Analyzing gage_flow_mgd.hdf5...")
    gage_file = os.path.join(INPUT_DIR, "gage_flow_mgd.hdf5")
    with h5py.File(gage_file, "r") as f:
        sample = list(f.keys())[0]
        real_ids = list(f[sample].attrs["column_labels"])

        gage_matrix_rows = []
        for node in REGRESSION_NODES:
            row = [pct_near_zero(f[node][r][:]) for r in real_ids]
            gage_matrix_rows.append(row)

    gage_matrix = np.array(gage_matrix_rows)
    make_heatmap(gage_matrix, REGRESSION_NODES,
                 "Gage flow (MGD): % values ≤ 0.01",
                 "zero_pct_gage_flow.png")

    # ----------------------------------------------------------------
    # 3. Catchment inflow HDF5
    # ----------------------------------------------------------------
    print("3. Analyzing catchment_inflow_mgd.hdf5...")
    catch_file = os.path.join(INPUT_DIR, "catchment_inflow_mgd.hdf5")
    with h5py.File(catch_file, "r") as f:
        sample = list(f.keys())[0]
        catch_real_ids = list(f[sample].attrs["column_labels"])

        catch_matrix_rows = []
        for node in REGRESSION_NODES:
            row = [pct_near_zero(f[node][r][:]) for r in catch_real_ids]
            catch_matrix_rows.append(row)

    catch_matrix = np.array(catch_matrix_rows)
    make_heatmap(catch_matrix, REGRESSION_NODES,
                 "Catchment inflow (MGD): % values ≤ 0.01\n(after upstream subtraction; x = ≥99.5%)",
                 "zero_pct_catchment_inflow.png")

    # ----------------------------------------------------------------
    # Summary stats
    # ----------------------------------------------------------------
    print("\n=== Summary: nodes with >90% zeros in ANY realization ===")
    for i, node in enumerate(REGRESSION_NODES):
        max_pct = catch_matrix[i].max()
        n_high = np.sum(catch_matrix[i] > 90)
        if n_high > 0:
            print(f"  {node:25s}: {n_high:3d} realizations >90% zero (max {max_pct:.1f}%)")


if __name__ == "__main__":
    main()
