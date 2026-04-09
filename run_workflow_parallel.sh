#!/bin/bash
#SBATCH --job-name=run_ensemble
#SBATCH --output=./logs/run2.out
#SBATCH --error=./logs/run2.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=35


# Setup
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Workflow flags
CSV_TO_HDF=${CSV_TO_HDF:-true}
CALC_INFLOW=${CALC_INFLOW:-true}
PREP=${PREP:-true}
SIMULATE=${SIMULATE:-true}

# Create directories (dataset-specific paths are under pywrdrb/<flow_type>/)
PYWRDRB_DIR=$(python3 -c "from config import PYWRDRB_DIR; print(PYWRDRB_DIR)")
FLOW_TYPE=$(python3 -c "from config import FLOW_TYPE; print(FLOW_TYPE)")
mkdir -p logs "$PYWRDRB_DIR"/{inputs,outputs,models}
echo "Dataset: $FLOW_TYPE"
echo "Output dir: $PYWRDRB_DIR"


# Execute workflow
[ "$CSV_TO_HDF" = true ] && python3 01_csv_to_hdf.py
[ "$CALC_INFLOW" = true ] && python3 02_calc_catchment_inflow.py
[ "$PREP" = true ] && mpirun -np $np python3 03_prepare_inputs.py
[ "$SIMULATE" = true ] && mpirun -np $np python3 04_run_simulations_parallel.py

echo "Workflow complete."