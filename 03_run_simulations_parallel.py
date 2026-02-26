"""Build, load, and run pywrdrb simulations using ensemble inflow data."""
import os
import re
import glob
import math
import numpy as np
import pywrdrb
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers, combine_batched_hdf5_outputs

USE_MPI = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "pywrdrb", "inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "pywrdrb", "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "pywrdrb", "models")
CATCHMENT_INFLOW_FILE = os.path.join(INPUT_DIR, "catchment_inflow_mgd.hdf5")

FLOW_TYPE = "sandeep_hybrid"
START_DATE = "1945-01-01"
END_DATE = "2018-12-31"
N_REALIZATIONS_PER_BATCH = 5

SAVE_RESULTS_SETS = [
    "major_flow", "inflow", "res_storage",
    "lower_basin_mrf_contributions", "mrf_target",
    "ibt_diversions", "ibt_demands",
    "nyc_release_components", "res_level",
]


def get_parameter_subset_to_export(all_parameter_names, results_set_subset):
    output_loader = pywrdrb.load.Output(output_filenames=[])
    keep_keys = []
    for results_set in results_set_subset:
        if results_set == "all":
            continue
        keys_subset, _ = output_loader.get_keys_and_column_names_for_results_set(
            all_parameter_names, results_set
        )
        keep_keys.extend(keys_subset)
    return keep_keys


def register_flow_type():
    pn_config = pywrdrb.get_pn_config()
    pn_config[f"flows/{FLOW_TYPE}"] = os.path.abspath(INPUT_DIR)
    pywrdrb.load_pn_config(pn_config)


def run_simulations():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    if USE_MPI:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1

    register_flow_type()

    # Get realization IDs
    if rank == 0:
        realization_ids = get_hdf5_realization_numbers(CATCHMENT_INFLOW_FILE)
        realization_ids = [str(r) for r in realization_ids]
        print(f"Found {len(realization_ids)} realizations")
    else:
        realization_ids = None

    if USE_MPI and comm:
        realization_ids = comm.bcast(realization_ids, root=0)

    # Distribute across ranks
    rank_realization_ids = list(np.array_split(realization_ids, size)[rank])
    n_rank_realizations = len(rank_realization_ids)

    if n_rank_realizations == 0:
        print(f"Rank {rank}: No realizations assigned")
        return

    # Split into batches
    n_batches = math.ceil(n_rank_realizations / N_REALIZATIONS_PER_BATCH)
    print(f"Rank {rank}: {n_rank_realizations} realizations, {n_batches} batches")

    # Clean old batch files
    if rank == 0:
        for pattern in [
            os.path.join(OUTPUT_DIR, f"{FLOW_TYPE}_rank*_batch*.hdf5"),
            os.path.join(MODEL_DIR, f"{FLOW_TYPE}_rank*_batch*.json"),
        ]:
            for f in glob.glob(pattern):
                os.remove(f)

    if USE_MPI and comm:
        comm.Barrier()

    batch_filenames = []
    for batch in range(n_batches):
        batch_start = batch * N_REALIZATIONS_PER_BATCH
        batch_end = min((batch + 1) * N_REALIZATIONS_PER_BATCH, n_rank_realizations)
        indices = [str(r) for r in rank_realization_ids[batch_start:batch_end]]

        print(f"Rank {rank}, batch {batch + 1}/{n_batches}: {len(indices)} realizations")

        mb = pywrdrb.ModelBuilder(
            inflow_type=FLOW_TYPE,
            start_date=START_DATE,
            end_date=END_DATE,
            options={
                "inflow_ensemble_indices": indices,
                "nyc_nj_demand_source": "custom",
                "flow_prediction_mode": "regression_disagg",
            },
        )

        model_fname = os.path.join(MODEL_DIR, f"{FLOW_TYPE}_rank{rank}_batch{batch}.json")
        mb.make_model()
        mb.write_model(model_fname)

        model = pywrdrb.Model.load(model_fname)

        all_parameter_names = [p.name for p in model.parameters if p.name]
        subset_names = get_parameter_subset_to_export(all_parameter_names, SAVE_RESULTS_SETS)
        export_parameters = [p for p in model.parameters if p.name in subset_names]

        batch_output = os.path.join(OUTPUT_DIR, f"{FLOW_TYPE}_rank{rank}_batch{batch}.hdf5")
        pywrdrb.OutputRecorder(
            model=model,
            output_filename=batch_output,
            parameters=export_parameters,
        )

        model.run()
        batch_filenames.append(batch_output)
        del model

    # Combine batches
    if USE_MPI and comm:
        comm.Barrier()

    if rank == 0:
        batch_pattern = os.path.join(OUTPUT_DIR, f"{FLOW_TYPE}_rank*_batch*.hdf5")
        all_batch_files = glob.glob(batch_pattern)

        def sort_key(filename):
            match = re.search(r"rank(\d+)_batch(\d+)", filename)
            return (int(match.group(1)), int(match.group(2))) if match else (0, 0)

        all_batch_files.sort(key=sort_key)

        output_file = os.path.join(OUTPUT_DIR, f"{FLOW_TYPE}.hdf5")
        if os.path.exists(output_file):
            os.remove(output_file)

        print(f"Combining {len(all_batch_files)} batch files...")
        combine_batched_hdf5_outputs(all_batch_files, output_file)

        # Cleanup batch files
        for f in all_batch_files:
            os.remove(f)
        for f in glob.glob(os.path.join(MODEL_DIR, f"{FLOW_TYPE}_rank*_batch*.json")):
            os.remove(f)

        print(f"Done. Output: {output_file}")


if __name__ == "__main__":
    run_simulations()
