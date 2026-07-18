"""Global experiment configuration."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("NOVEL_BIO_ROOT", Path(__file__).resolve().parent.parent))

# Datasets
SWISSPROT_FASTA = ROOT / "datasets" / "swissprot" / "full_human_proteome.fasta"
PROTEINGYM_DIR = ROOT / "datasets" / "proteingym" / "ProteinGym_substitutions"
PROTEINGYM_REF = ROOT / "datasets" / "proteingym" / "DMS_substitutions.csv"

# Results
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
for p in (RESULTS, FIGURES, LOGS):
    p.mkdir(exist_ok=True, parents=True)

# Model
ESM_MODEL = "facebook/esm2_t6_8M_UR50D"
ESM_LAYER = 4
NUM_HIDDEN_LAYERS = 6
HIDDEN_DIM = 320
SAE_DICT_SIZE = 10240

# Sampling
N_PROTEINS = 1500
SEQ_LEN_MIN = 50
SEQ_LEN_MAX = 500
SEED = 42

# Annotation fetch
UNIPROT_BATCH = 50
UNIPROT_FIELDS = (
    "accession,id,sequence,"
    "ft_act_site,ft_binding,ft_disulfid,ft_motif,ft_domain,ft_region,"
    "ft_helix,ft_strand,ft_turn,ft_transmem,ft_signal,ft_chain,ft_mod_res,ft_lipid,ft_glyco,ft_carbohyd,ft_metal,ft_site"
)

# Feature analysis
FEATURE_ACTIVATION_PERCENTILE = 99.0  # threshold for "active" at residue
DARK_F1_THRESHOLD = 0.2
DARK_MIN_ACTIVE_RESIDUES = 100  # need this many activations to be considered structured

# Ablation
N_ABLATION_PROTEINS = 30
N_CONTROL_RESIDUES_PER_PROT = 10

# Device
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
