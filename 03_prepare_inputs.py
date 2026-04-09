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
FLOW_PREDICTION_MODES = ("perfect_foresight",)
SKIP_INFLOW_PREP = False

def register_flow_type():
    pn_config = pywrdrb.get_pn_config()
    pn_config[f"flows/{FLOW_TYPE}"] = os.path.abspath(INPUT_DIR)
    pywrdrb.load_pn_config(pn_config)


def main():
    # Initialize MPI once and share the communicator
    if USE_MPI:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
    else:
        comm = None
        rank = 0

    # Register flow type with pywrdrb
    register_flow_type()

    realization_ids = get_hdf5_realization_numbers(CATCHMENT_INFLOW_FILE)
    
    if rank == 0:
        print(f"Preparing inputs for {len(realization_ids)} realizations")

    if not SKIP_INFLOW_PREP:
        if rank == 0:
            print("Processing predicted inflows...")
        inflow_pre = PredictedInflowEnsemblePreprocessor(
            flow_type=FLOW_TYPE,
            ensemble_hdf5_file=CATCHMENT_INFLOW_FILE,
            realization_ids=realization_ids,
            start_date=None,
            end_date=None,
            modes=FLOW_PREDICTION_MODES,
            use_log=True,
            remove_zeros=True,
            use_const=False,
            use_mpi=USE_MPI,
            comm=comm,
        )
        inflow_pre.load()
        inflow_pre.process()
        inflow_pre.save()
        del inflow_pre
    else:
        if rank == 0:
            print("Skipping inflow preprocessing (SKIP_INFLOW_PREP=True)")

    # NJ diversions
    if rank == 0:
        print("Processing NJ diversions...")
    nj_pre = ExtrapolatedDiversionEnsemblePreprocessor(
        loc="nj",
        flow_type=FLOW_TYPE,
        ensemble_hdf5_file=GAGE_FLOW_FILE,
        realization_ids=realization_ids,
        use_mpi=USE_MPI,
        comm=comm,
    )
    nj_pre.load()
    nj_pre.process()
    nj_pre.save()
    del nj_pre

    # NYC diversions
    if rank == 0:
        print("Processing NYC diversions...")
    nyc_pre = ExtrapolatedDiversionEnsemblePreprocessor(
        loc="nyc",
        flow_type=FLOW_TYPE,
        ensemble_hdf5_file=GAGE_FLOW_FILE,
        realization_ids=realization_ids,
        use_mpi=USE_MPI,
        comm=comm,
    )
    nyc_pre.load()
    nyc_pre.process()
    nyc_pre.save()
    del nyc_pre

    # Predicted diversions
    if rank == 0:
        print("Processing predicted diversions...")
    nj_div_file = os.path.join(INPUT_DIR, "diversion_nj_extrapolated_mgd.hdf5")
    div_pre = PredictedDiversionEnsemblePreprocessor(
        flow_type=FLOW_TYPE,
        ensemble_hdf5_file=nj_div_file,
        realization_ids=realization_ids,
        start_date=None,
        end_date=None,
        modes=FLOW_PREDICTION_MODES,
        use_log=True,
        remove_zeros=True,
        use_const=False,
        use_mpi=USE_MPI,
        comm=comm,
    )
    div_pre.load()
    div_pre.process()
    div_pre.save()
    del div_pre
    
    if rank == 0:
        print("All inputs prepared.")


if __name__ == "__main__":
    main()
