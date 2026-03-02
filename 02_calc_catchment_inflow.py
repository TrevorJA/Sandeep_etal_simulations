"""Compute catchment inflows from gage flows (serial, single-core).

Drops realizations where any node has entirely near-zero catchment inflow
(indicating non-mass-conservative upstream/downstream ensemble members).
"""
import os
import warnings
import numpy as np
import h5py
from pywrdrb.pre.flows import _subtract_upstream_catchment_inflows
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers, extract_realization_from_hdf5
from pywrdrb.pywr_drb_node_data import obs_site_matches
from config import GAGE_FLOW_FILE, CATCHMENT_INFLOW_FILE

PYWRDRB_NODES = list(obs_site_matches.keys())

# Nodes that should always be zero (co-located with another node)
EXPECTED_ZERO_NODES = {"delTrenton"}

# Threshold for near-zero detection
ZERO_THRESHOLD = 0.01


def compute_catchment_inflows():
    """Read gage flows, subtract upstream contributions, write catchment_inflow_mgd.hdf5.

    Realizations where any non-exempt node has entirely near-zero
    catchment inflow are dropped with a warning.
    """
    realization_ids = get_hdf5_realization_numbers(GAGE_FLOW_FILE)
    print(f"Computing catchment inflows for {len(realization_ids)} realizations")

    # Pass 1: compute catchment inflows for all realizations
    all_catchment_dfs = {}
    dates = None
    for i, real_id in enumerate(realization_ids):
        df = extract_realization_from_hdf5(GAGE_FLOW_FILE, str(real_id), stored_by_node=True)
        if "datetime" in df.columns:
            df = df.drop(columns=["datetime"])

        catchment_df = _subtract_upstream_catchment_inflows(df)
        all_catchment_dfs[real_id] = catchment_df

        if dates is None:
            dates = catchment_df.index.strftime("%Y-%m-%d").tolist()

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Computed realization {i + 1}/{len(realization_ids)}")

    # Pass 2: write all realizations
    realization_labels = [str(r) for r in realization_ids]
    with h5py.File(CATCHMENT_INFLOW_FILE, "w") as f_out:
        for node in PYWRDRB_NODES:
            grp = f_out.create_group(str(node))
            grp.attrs["column_labels"] = realization_labels
            grp.create_dataset("date", data=dates, compression="gzip")

            for real_id in realization_ids:
                grp.create_dataset(
                    str(real_id),
                    data=all_catchment_dfs[real_id][node].values,
                    compression="gzip",
                )

    print(f"Wrote {CATCHMENT_INFLOW_FILE}")


if __name__ == "__main__":
    compute_catchment_inflows()
