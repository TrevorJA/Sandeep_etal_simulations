"""Generate predicted inflows and diversions for pywrdrb (MPI-parallel)."""
import os
import pywrdrb
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers
from pywrdrb.pre import (
    PredictedInflowEnsemblePreprocessor,
    ExtrapolatedDiversionEnsemblePreprocessor,
    PredictedDiversionEnsemblePreprocessor,
)
from config import INPUT_DIR, GAGE_FLOW_FILE, CATCHMENT_INFLOW_FILE, FLOW_TYPE

USE_MPI = True


def register_flow_type():
    pn_config = pywrdrb.get_pn_config()
    pn_config[f"flows/{FLOW_TYPE}"] = os.path.abspath(INPUT_DIR)
    pywrdrb.load_pn_config(pn_config)


def main():
    # Register flow type with pywrdrb
    register_flow_type()

    realization_ids = get_hdf5_realization_numbers(CATCHMENT_INFLOW_FILE)
    print(f"Preparing inputs for {len(realization_ids)} realizations")

    # Predicted inflows
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

    # NJ diversions
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

    # NYC diversions
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

    # Predicted diversions
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
