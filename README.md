# Sandeep et al. Simulations

Pywrdrb simulations using hybrid fine-tuned ensemble streamflow data (160 members, 1915-2018).

## Requirements

- [Pywr-DRB](https://github.com/Pywr-DRB/Pywr-DRB) installed (`pywrdrb` package)
- Ensemble CSVs in `drb_streamflow_ensembles/hybrid_finetuned/`
- `basin_attributes.csv` (drainage areas for unit conversion)

## Usage

Install dependencies (pywrdrb).  On Hopper:

```bash
# setup env
module load python/3.11.5
python3 -m venv venv
source venv/bin/activate

# install pywrdrb
pip install git+https://github.com/Pywr-DRB/Pywr-DRB.git
```


Run the full workflow:

```bash
sbatch run_workflow_parallel.sh
```


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
