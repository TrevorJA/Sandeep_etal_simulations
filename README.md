# Sandeep et al. Simulations

Pywr-DRB ensemble simulations using streamflow ensembles from differentiable and LSTM models.

## Datasets

Configured via `ENSEMBLE_DATASET` in `config.py`:

| Dataset | Members | Flow Type |
|---------|---------|-----------|
| `differentiable_model` | 160 | `ensemble_differentiable_model` |
| `lstm_model` | 128 | `ensemble_lstm_model` |

## Requirements

- [Pywr-DRB](https://github.com/Pywr-DRB/Pywr-DRB) (`add_perfect_foresight_mode` branch)
- Ensemble CSVs in `drb_streamflow_ensembles/<dataset>/`
- `basin_attributes.csv` (drainage areas for unit conversion)

## Setup

On Hopper:

```bash
module load python/3.11.5
python3 -m venv venv
source venv/bin/activate

# install pywrdrb 
# IMPORTANT: this data only works with the add_perfect_foresight_mode branch
pip install "git+https://github.com/Pywr-DRB/Pywr-DRB.git@add_perfect_foresight_mode"
```

## Usage

Run the full workflow:

```bash
sbatch run_workflow_parallel.sh
```

Individual workflow steps can be toggled via environment variables:

```bash
# Run only preprocessing (skip simulation)
sbatch --export=ALL,SIMULATE=false run_workflow_parallel.sh

# Run only simulation (inputs already exist)
sbatch --export=ALL,PREP=false run_workflow_parallel.sh

# Regenerate gage flow HDF5 from CSVs
sbatch --export=ALL,CSV_TO_HDF=true run_workflow_parallel.sh
```

## Workflow Steps

1. `01_csv_to_hdf.py` - Convert ensemble CSVs to HDF5 gage flow file
2. `02_calc_catchment_inflow.py` - Calculate catchment inflows from gage flows
3. `03_prepare_inputs.py` - Generate predicted inflows, diversions (MPI-parallel)
4. `04_run_simulations_parallel.py` - Run Pywr-DRB simulations (MPI-parallel)

## MPI Notes

- Uses single-node MPI (`--nodes=1`, 35 ranks) to avoid cross-node OpenMPI communication issues on Hopper.
- Scripts 03 and 04 are MPI-parallel; 01 and 02 run serially.

## Directory Structure

```
pywrdrb/<flow_type>/
  inputs/          # HDF5 input files (gage flows, catchment inflows, diversions, predictions)
  models/          # Model JSON files (temporary, cleaned after run)
  outputs/         # Simulation output HDF5
figures/<flow_type>/  # Diagnostic and analysis figures
```
