"""Compute catchment inflows from gage flows (serial, single-core)."""
import os
import h5py
from pywrdrb.pre.flows import _subtract_upstream_catchment_inflows
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers, extract_realization_from_hdf5
from pywrdrb.pywr_drb_node_data import obs_site_matches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "pywrdrb", "inputs")
GAGE_FLOW_FILE = os.path.join(INPUT_DIR, "gage_flow_mgd.hdf5")
CATCHMENT_INFLOW_FILE = os.path.join(INPUT_DIR, "catchment_inflow_mgd.hdf5")

PYWRDRB_NODES = list(obs_site_matches.keys())


def compute_catchment_inflows():
    """Read gage flows, subtract upstream contributions, write catchment_inflow_mgd.hdf5."""
    realization_ids = get_hdf5_realization_numbers(GAGE_FLOW_FILE)
    print(f"Computing catchment inflows for {len(realization_ids)} realizations")

    with h5py.File(CATCHMENT_INFLOW_FILE, "w") as f_out:
        for i, real_id in enumerate(realization_ids):
            # Extract one realization as DataFrame (columns=nodes, index=dates)
            df = extract_realization_from_hdf5(GAGE_FLOW_FILE, str(real_id), stored_by_node=True)
            # Drop datetime column if present
            if "datetime" in df.columns:
                df = df.drop(columns=["datetime"])

            # Subtract upstream flows
            catchment_df = _subtract_upstream_catchment_inflows(df)

            if i == 0:
                # First realization: set up node groups with date dataset
                dates = catchment_df.index.strftime("%Y-%m-%d").tolist()
                realization_labels = [str(r) for r in realization_ids]
                for node in PYWRDRB_NODES:
                    grp = f_out.create_group(str(node))
                    grp.attrs["column_labels"] = realization_labels + ["date"]
                    grp.create_dataset("date", data=dates, compression="gzip")

            # Write this realization's data for each node
            for node in PYWRDRB_NODES:
                f_out[str(node)].create_dataset(
                    str(real_id), data=catchment_df[node].values, compression="gzip"
                )

            if (i + 1) % 20 == 0 or i == 0:
                print(f"  Processed realization {i + 1}/{len(realization_ids)}")

    print(f"Wrote {CATCHMENT_INFLOW_FILE}")


if __name__ == "__main__":
    compute_catchment_inflows()
