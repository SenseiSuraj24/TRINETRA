"""
config.py — AURA Global Configuration
======================================
Single source of truth for all hyperparameters, paths, and system constants.
Centralising config prevents magic numbers from scattering across modules and
makes hackathon tuning fast (one file to change).
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.resolve()
CSV_DIR    = BASE_DIR / "CSV's" / "MachineLearningCVE"
MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR   = BASE_DIR / "logs"
CONTRACTS_DIR = BASE_DIR / "contracts"

# Ensure output dirs exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

# The MachineLearningCSV variant strips IPs → we map flows to synthetic nodes.
# NUM_SYNTHETIC_NODES simulates the number of distinct IP endpoints in the org.
NUM_SYNTHETIC_NODES = 20

# Column name for the target label (has a leading space in the CSV)
LABEL_COL = " Label"

# The label value that represents benign (normal) traffic in CICIDS2017
BENIGN_LABEL = "BENIGN"

# Fraction of data to load per CSV (1.0 = all rows; reduce for speed during dev)
DATA_LOAD_FRACTION = 0.3   # 30 % is enough to demo; use 1.0 for full training

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH / TTL EDGE DECAY
# ─────────────────────────────────────────────────────────────────────────────

# Rolling time-window size in simulated "ticks" (1 tick ≈ 1 second of NetFlow)
WINDOW_SIZE = 60          # number of flow rows per graph snapshot

# Time-To-Live: an edge is pruned after this many windows without traffic
EDGE_TTL_WINDOWS = 3

# ─────────────────────────────────────────────────────────────────────────────
# AUTOENCODER (Layer 1 — Statistical Tripwire)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DIM     = 78    # Number of normalised NetFlow statistical features
ENCODER_DIMS    = [64, 32]   # Progressive compression (avoid gradient explosion)
LATENT_DIM      = 16    # Bottleneck: the latent fingerprint space
DECODER_DIMS    = [32, 64]   # Mirror of encoder (symmetric reconstruction)

AE_LEARNING_RATE = 1e-3
AE_EPOCHS        = 30        # Enough for convergence on CICIDS2017 subset
AE_BATCH_SIZE    = 256

# Contrastive negative-sampling margin (pushes attack embeddings away from
# the normal manifold during simulated baseline hardening)
CONTRASTIVE_MARGIN = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# STGNN (Layer 2 — Contextual Validator)
# ─────────────────────────────────────────────────────────────────────────────

# Node feature dimensionality fed to the GNN
# Each node's feature vector = mean of its incident edge (flow) features
GNN_INPUT_DIM  = FEATURE_DIM
GNN_HIDDEN_DIM = 64
GNN_OUTPUT_DIM = 32          # Latent node embedding dimension
GNN_LEARNING_RATE = 5e-4
GNN_EPOCHS     = 20

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC THRESHOLDING (Exponential Moving Average over batch MSE)
# ─────────────────────────────────────────────────────────────────────────────

# EMA smoothing factor (α). Higher = reacts faster but is noisier.
# Lower = more stable but slower to adapt.
EMA_ALPHA = 0.05

# An alert is raised when:  loss > EMA_mean + (EMA_SIGMA_MULTIPLIER × EMA_std)
EMA_SIGMA_MULTIPLIER = 3.0

# Warm-up batches before thresholds are active (avoids cold-start false alarms)
EMA_WARMUP_BATCHES = 50

# ─────────────────────────────────────────────────────────────────────────────
# FEDERATED LEARNING (Flower + Krum Aggregation)
# ─────────────────────────────────────────────────────────────────────────────

FL_SERVER_ADDRESS   = "localhost:8080"
FL_NUM_ROUNDS       = 3          # 3 rounds for 3 clients — 1 hash per round on ledger
FL_MIN_CLIENTS      = 2          # Minimum clients needed to start a round
FL_MIN_AVAILABLE    = 2

# Krum: number of clients to select per round (must be ≤ total clients - 2)
# Krum drops the m clients whose weight updates are most distant from the median.
KRUM_NUM_TO_SELECT  = 2          # Select 2 from 3 mock clients (drops 1 straggler)

# Straggler policy: if a client doesn't respond within this many seconds, drop it
FL_ROUND_TIMEOUT_SEC = 30

# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE ENGINE — Critical Infrastructure Allowlist
# ─────────────────────────────────────────────────────────────────────────────

# Tier-1 "never hard-isolate" nodes (simulated by synthetic node IDs)
# In production these would be real IP CIDRs or hostnames.
CRITICAL_ALLOWLIST = {
    "node_0":  "Domain Controller (AD)",
    "node_1":  "Core HR Database",
    "node_2":  "Payment Gateway",
    "node_3":  "SCADA / ICS Controller",
}

# Confidence thresholds for the 3-tier response policy
CONFIDENCE_LOW_THRESHOLD  = 0.40   # Below this: log only
CONFIDENCE_MED_THRESHOLD  = 0.70   # Below this: throttle + HITL
# Above MED_THRESHOLD → full isolation for non-critical nodes

# ─────────────────────────────────────────────────────────────────────────────
# BLOCKCHAIN / GANACHE (Immutable Audit Log)
# ─────────────────────────────────────────────────────────────────────────────

GANACHE_URL              = "http://127.0.0.1:7545"
CONTRACT_ADDRESS_FILE    = str(MODELS_DIR / "contract_address.txt")
CONTRACT_ABI_FILE        = str(CONTRACTS_DIR / "ModelRegistry.abi")

# If Ganache is not running, AURA falls back to local SHA-256 file logging
BLOCKCHAIN_FALLBACK_LOG  = str(LOGS_DIR / "blockchain_fallback.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# ISOLATION FOREST (Baseline Sanitisation)
# ─────────────────────────────────────────────────────────────────────────────

# Contamination: expected fraction of mislabelled / poisoned rows in the
# "normal" training split.  CICIDS2017 Monday CSV is ~99.9% benign but we
# apply a small contamination rate defensively.
IF_CONTAMINATION = 0.02   # 2 % — removes extreme statistical outliers

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD_REFRESH_INTERVAL_MS = 1500   # Streamlit auto-refresh period
ALERT_LOG_FILE = str(LOGS_DIR / "aura_alerts.jsonl")
EVENT_LOG_FILE = str(LOGS_DIR / "aura_events.jsonl")
