"""Shared configuration for the ensemble simulation workflow."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Dataset selection ----
# Options: "hybrid_finetuned" (160 members), "lstm_model" (128 members)
ENSEMBLE_DATASET = "lstm_model"

DATASET_CONFIG = {
    "hybrid_finetuned": {"n_realizations": 160, "flow_type": "ensemble_hybrid_finetuned"},
    "lstm_model": {"n_realizations": 128, "flow_type": "ensemble_lstm_model"},
}

N_REALIZATIONS = DATASET_CONFIG[ENSEMBLE_DATASET]["n_realizations"]
FLOW_TYPE = DATASET_CONFIG[ENSEMBLE_DATASET]["flow_type"]

# ---- Paths ----
CSV_DIR = os.path.join(BASE_DIR, "drb_streamflow_ensembles", ENSEMBLE_DATASET)
PYWRDRB_DIR = os.path.join(BASE_DIR, "pywrdrb", FLOW_TYPE)
INPUT_DIR = os.path.join(PYWRDRB_DIR, "inputs")
OUTPUT_DIR = os.path.join(PYWRDRB_DIR, "outputs")
MODEL_DIR = os.path.join(PYWRDRB_DIR, "models")
BASIN_ATTRS_FILE = os.path.join(BASE_DIR, "basin_attributes.csv")

GAGE_FLOW_FILE = os.path.join(INPUT_DIR, "gage_flow_mgd.hdf5")
CATCHMENT_INFLOW_FILE = os.path.join(INPUT_DIR, "catchment_inflow_mgd.hdf5")

# ---- Figures ----
FIG_DIR = os.path.join(BASE_DIR, "figures", FLOW_TYPE)
os.makedirs(FIG_DIR, exist_ok=True)
