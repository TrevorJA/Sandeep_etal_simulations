# Sandeep et al. Simulations

Pywrdrb simulations using hybrid fine-tuned ensemble streamflow data (160 members, 1915-2018).

## Requirements

- [Pywr-DRB](https://github.com/Pywr-DRB/Pywr-DRB) installed (`pywrdrb` package)
- Ensemble CSVs in `drb_streamflow_ensembles/hybrid_finetuned/`
- `basin_attributes.csv` (drainage areas for unit conversion)

## Usage

Run scripts in order:

```bash
# 1. Convert CSVs to HDF5 (mm/day -> MGD)
python 01_csv_to_hdf.py

# 2. Generate catchment inflows, predicted inflows, and diversions
python 02_prepare_inputs.py

# 3. Run pywrdrb simulations
python 03_run_simulations_parallel.py
```

Set `USE_MPI = True` in scripts 02/03 for MPI parallelism (`mpirun -n N python ...`).

## Directory Structure

```
pywrdrb/
├── inputs/          # HDF5 input files (gage flows, catchment inflows, diversions)
├── models/          # Model JSON files (temporary, cleaned after run)
└── outputs/         # Simulation output HDF5
```

## Outputs

- `pywrdrb/inputs/gage_flow_mgd.hdf5`
- `pywrdrb/inputs/catchment_inflow_mgd.hdf5`
- `pywrdrb/inputs/predicted_inflows_mgd.hdf5`
- `pywrdrb/inputs/diversion_{nyc,nj}_extrapolated_mgd.hdf5`
- `pywrdrb/inputs/predicted_diversions_mgd.hdf5`
- `pywrdrb/outputs/sandeep_hybrid.hdf5`
