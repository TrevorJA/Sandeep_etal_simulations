#!/bin/bash
#SBATCH --job-name=run_ensemble
#SBATCH --output=./logs/run.out
#SBATCH --error=./logs/run.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=35


# Setup
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Workflow flags
CSV_TO_HDF=${CSV_TO_HDF:-true}
PREP=${PREP:-true}
SIMULATE=${SIMULATE:-true}

# Create directories
mkdir -p logs pywrdrb/{inputs,outputs,models} figures


# Execute workflow
[ "$PREP" = true ] && python3 01_csv_to_hdf.py
[ "$PREP" = true ] && python3 02_calc_catchment_inflow.py
[ "$PREP" = true ] && mpirun -np $np python3 03_prepare_inputs.py
[ "$SIMULATE" = true ] && mpirun -np $np python3 04_run_simulations_parallel.py

echo "Workflow complete."