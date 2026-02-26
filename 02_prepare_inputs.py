"""Generate catchment inflow HDF5, predicted inflows, and diversions for pywrdrb."""
import os
import h5py
import pandas as pd
import pywrdrb
from pywrdrb.pre.flows import _subtract_upstream_catchment_inflows
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers, extract_realization_from_hdf5
from pywrdrb.pre import (
    PredictedInflowEnsemblePreprocessor,
    ExtrapolatedDiversionEnsemblePreprocessor,
    PredictedDiversionEnsemblePreprocessor,
)
from pywrdrb.pywr_drb_node_data import obs_site_matches

USE_MPI = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "pywrdrb", "inputs")
GAGE_FLOW_FILE = os.path.join(INPUT_DIR, "gage_flow_mgd.hdf5")
CATCHMENT_INFLOW_FILE = os.path.join(INPUT_DIR, "catchment_inflow_mgd.hdf5")
FLOW_TYPE = "sandeep_hybrid"

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


def register_flow_type():
    pn_config = pywrdrb.get_pn_config()
    pn_config[f"flows/{FLOW_TYPE}"] = os.path.abspath(INPUT_DIR)
    pywrdrb.load_pn_config(pn_config)


def main():
    # Step 1: Compute catchment inflows from gage flows
    compute_catchment_inflows()

    # Step 2: Register flow type with pywrdrb
    register_flow_type()

    realization_ids = get_hdf5_realization_numbers(CATCHMENT_INFLOW_FILE)
    print(f"Preparing inputs for {len(realization_ids)} realizations")

    # Step 3: Predicted inflows
    print("Processing predicted inflows...")
    inflow_pre = PredictedInflowEnsemblePreprocessor(
        flow_type=FLOW_TYPE,
        ensemble_hdf5_file=CATCHMENT_INFLOW_FILE,
        realization_ids=realization_ids,
        start_date=None,
        end_date=None,
        modes=("regression_disagg",),
        use_log=True,
        remove_zeros=True,
        use_const=False,
        use_mpi=USE_MPI,
    )
    inflow_pre.load()
    inflow_pre.process()
    inflow_pre.save()
    del inflow_pre
    print("  Done.")

    # Step 4: NJ diversions
    print("Processing NJ diversions...")
    nj_pre = ExtrapolatedDiversionEnsemblePreprocessor(
        loc="nj",
        flow_type=FLOW_TYPE,
        ensemble_hdf5_file=GAGE_FLOW_FILE,
        realization_ids=realization_ids,
        use_mpi=USE_MPI,
    )
    nj_pre.load()
    nj_pre.process()
    nj_pre.save()
    del nj_pre
    print("  Done.")

    # Step 5: NYC diversions
    print("Processing NYC diversions...")
    nyc_pre = ExtrapolatedDiversionEnsemblePreprocessor(
        loc="nyc",
        flow_type=FLOW_TYPE,
        ensemble_hdf5_file=GAGE_FLOW_FILE,
        realization_ids=realization_ids,
        use_mpi=USE_MPI,
    )
    nyc_pre.load()
    nyc_pre.process()
    nyc_pre.save()
    del nyc_pre
    print("  Done.")

    # Step 6: Predicted diversions
    print("Processing predicted diversions...")
    nj_div_file = os.path.join(INPUT_DIR, "diversion_nj_extrapolated_mgd.hdf5")
    div_pre = PredictedDiversionEnsemblePreprocessor(
        flow_type=FLOW_TYPE,
        ensemble_hdf5_file=nj_div_file,
        realization_ids=realization_ids,
        start_date=None,
        end_date=None,
        modes=("regression_disagg",),
        use_log=True,
        remove_zeros=True,
        use_const=False,
        use_mpi=USE_MPI,
    )
    div_pre.load()
    div_pre.process()
    div_pre.save()
    del div_pre
    print("  Done.")

    print("All inputs prepared.")


if __name__ == "__main__":
    main()
