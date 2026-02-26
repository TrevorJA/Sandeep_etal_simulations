"""Convert ensemble CSV streamflow data to pywrdrb HDF5 format (gage_flow_mgd.hdf5)."""
import os
import numpy as np
import pandas as pd
import h5py
from pywrdrb.pywr_drb_node_data import obs_site_matches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "drb_streamflow_ensembles", "hybrid_finetuned")
INPUT_DIR = os.path.join(BASE_DIR, "pywrdrb", "inputs")
OUTPUT_FILE = os.path.join(INPUT_DIR, "gage_flow_mgd.hdf5")
BASIN_ATTRS_FILE = os.path.join(BASE_DIR, "basin_attributes.csv")

# mm/day over 1 km2 -> MGD
MM_PER_DAY_KM2_TO_MGD = 0.264172

PYWRDRB_NODES = list(obs_site_matches.keys())
N_REALIZATIONS = 160


def load_drainage_areas():
    df = pd.read_csv(BASIN_ATTRS_FILE)
    df["name"] = df["name"].astype(str)
    return dict(zip(df["name"], df["drainage_area"]))


def read_node_csv(node):
    fname = os.path.join(CSV_DIR, f"pred_{node}.csv")
    df = pd.read_csv(fname, index_col="date", parse_dates=True)
    sim_cols = [f"qsim_{i}" for i in range(1, N_REALIZATIONS + 1)]
    df = df[sim_cols]
    df.columns = [str(i) for i in range(N_REALIZATIONS)]
    return df


def main():
    areas = load_drainage_areas()

    missing_csv = [n for n in PYWRDRB_NODES if not os.path.exists(os.path.join(CSV_DIR, f"pred_{n}.csv"))]
    missing_area = [n for n in PYWRDRB_NODES if str(n) not in areas]
    if missing_csv:
        print(f"WARNING: Missing CSVs for nodes: {missing_csv}")
    if missing_area:
        print(f"WARNING: Missing drainage areas for nodes: {missing_area}")

    nodes_to_process = [n for n in PYWRDRB_NODES if n not in missing_csv and str(n) not in missing_area]
    realization_labels = [str(i) for i in range(N_REALIZATIONS)]

    print(f"Processing {len(nodes_to_process)} nodes, {N_REALIZATIONS} realizations")

    os.makedirs(INPUT_DIR, exist_ok=True)
    with h5py.File(OUTPUT_FILE, "w") as f:
        for node in nodes_to_process:
            df = read_node_csv(node)
            area = areas[str(node)]
            df = df * area * MM_PER_DAY_KM2_TO_MGD
            df = df.fillna(0.0)

            grp = f.create_group(str(node))
            grp.attrs["column_labels"] = realization_labels + ["date"]

            dates = df.index.strftime("%Y-%m-%d").tolist()
            grp.create_dataset("date", data=dates, compression="gzip")

            for col in realization_labels:
                grp.create_dataset(col, data=df[col].values, compression="gzip")

            print(f"  {node}: {len(dates)} days, area={area:.1f} km2")

    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
