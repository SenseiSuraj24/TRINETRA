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
# SEVERITY ENGINE — Temporal Accumulator + EMA Trajectory
# ─────────────────────────────────────────────────────────────────────────────

# Rolling window for per-node event accumulation (seconds).
# Events older than this are purged before escalation rules are evaluated.
TEMPORAL_WINDOW_SECONDS = 300   # 5 minutes — configurable

# EMA trajectory persistence threshold.
# K consecutive readings above 2.0σ → MEDIUM floor.
# K consecutive readings above 2.5σ → HIGH floor.
K_CONSECUTIVE_READINGS  = 5     # configurable


# ─────────────────────────────────────────────────────────────────────────────
# FEDERATED LEARNING (Flower + Krum Aggregation)
# ─────────────────────────────────────────────────────────────────────────────

FL_SERVER_ADDRESS   = "localhost:8080"
FL_NUM_ROUNDS       = 3          # 3 rounds for 3 clients — 1 hash per round on ledger
FL_MIN_CLIENTS      = 3          # Minimum clients needed to start a round
FL_MIN_AVAILABLE    = 3          # All 3 orgs must be present before round 1

# Krum: number of clients to select per round (must be ≤ total clients - 2)
# Krum drops the m clients whose weight updates are most distant from the median.
KRUM_NUM_TO_SELECT  = 2          # Select 2 from 3 mock clients (drops 1 straggler)

# Straggler policy: if a client doesn't respond within this many seconds, drop it
FL_ROUND_TIMEOUT_SEC = 30

# ─────────────────────────────────────────────────────────────────────────────
# FLTRUST AGGREGATION (replaces Krum — Upgrade 6)
# ─────────────────────────────────────────────────────────────────────────────

# Number of synthetic benign samples the server holds as its trusted root dataset.
# These are used to train the server model by one step each round so it computes
# a reference gradient direction for cosine trust scoring of client updates.
# Range: 100–500 recommended; lower = faster, higher = more robust server gradient.
FLTRUST_ROOT_SAMPLES   = 200

# Learning rate used for the server's single-step root-dataset gradient update.
# Kept separate from AE_LEARNING_RATE so it can be tuned independently.
FLTRUST_SERVER_LR      = 1e-3

# Trust score at or below this value causes the client to be flagged as Byzantine
# in the detection log (fed into Upgrade 3).  ReLU already zeroes negatives;
# this threshold lets you also zero out near-zero trust scores from noisy clients.
FLTRUST_MIN_TRUST_SCORE = 0.0   # 0.0 = ReLU only (strict); raise to e.g. 0.05 to be stricter

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

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM INJECTION CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# NetFlow feature index lookup — used by _run_inject_inference() in api_server.py
# to resolve feature names to their column positions without hardcoding integers.
# Derived from CICIDS2017 schema (Label and IP columns stripped).
FEATURE_INDEX_MAP: dict = {
    "dest_port":            0,
    "flow_duration":        1,
    "fwd_packets":          2,
    "bwd_packets":          3,
    "fwd_bytes":            4,
    "bwd_bytes":            5,
    "fwd_pkt_len_max":      6,
    "fwd_pkt_len_min":      7,
    "fwd_pkt_len_mean":     8,
    "fwd_pkt_len_std":      9,
    "bwd_pkt_len_max":      10,
    "bwd_pkt_len_min":      11,
    "bwd_pkt_len_mean":     12,
    "bwd_pkt_len_std":      13,
    "flow_bytes_s":         14,
    "flow_pkts_s":          15,
    "flow_iat_mean":        16,
    "flow_iat_std":         17,
    "flow_iat_max":         18,
    "flow_iat_min":         19,
    "fwd_iat_total":        20,
    "fwd_psh_flags":        30,
    "pkt_len_std":          41,
    "pkt_len_var":          42,
    "syn_flag_count":       44,
    "rst_flag_count":       45,
    "psh_flag_count":       46,
    "ack_flag_count":       47,
    "subflow_fwd_bytes":    63,
    "subflow_bwd_bytes":    65,
    "idle_mean":            74,
    "idle_std":             75,
}

# MSE severity thresholds for custom injection events.
# These values are calibrated to the current AE's reconstruction error scale.
# Raise MSE_THRESHOLD_HIGH to require stronger anomaly evidence for HIGH tier.
MSE_THRESHOLD_HIGH   = 0.7   # MSE above this → AlertSeverity.HIGH
MSE_THRESHOLD_MEDIUM = 0.4   # MSE above this → AlertSeverity.MEDIUM  (else LOW)

# Corruption profiles for each simulated attack type.
# Each profile maps feature-group names to their corruption ranges:
#   {feature_key_from_FEATURE_INDEX_MAP: (lo, hi)}
# _run_inject_inference() applies each group in order and skips absent keys.
ATTACK_CORRUPTION_PROFILES: dict = {
    "ddos": {
        "flow_pkts_s":      (0.90, 0.99),
        "flow_bytes_s":     (0.88, 0.99),
        "flow_iat_mean":    (0.00, 0.03),   # near-zero = flood
        "flow_iat_std":     (0.00, 0.02),   # robotic regularity
        "syn_flag_count":   (0.80, 0.99),
        "ack_flag_count":   (0.01, 0.08),
        "fwd_packets":      (0.90, 0.99),
    },
    "lateral": {
        "flow_duration":    (0.50, 0.75),
        "fwd_packets":      (0.50, 0.65),
        "flow_iat_std":     (0.75, 0.95),   # high jitter = evasion
        "idle_mean":        (0.80, 0.98),   # beacon-like pauses
        "idle_std":         (0.03, 0.10),   # robotic regularity
        "psh_flag_count":   (0.60, 0.80),
    },
    "exfil": {
        "fwd_bytes":           (0.88, 0.99),
        "bwd_bytes":           (0.00, 0.06),
        "subflow_fwd_bytes":   (0.85, 0.99),
        "subflow_bwd_bytes":   (0.00, 0.04),
        "flow_iat_mean":       (0.40, 0.55),   # regulated pacing
        "flow_iat_std":        (0.00, 0.03),   # robotic timing
        "flow_duration":       (0.80, 0.98),
    },
    "port_scan": {
        "flow_duration":    (0.00, 0.03),
        "fwd_bytes":        (0.00, 0.04),
        "bwd_bytes":        (0.00, 0.03),
        "rst_flag_count":   (0.80, 0.99),
        "syn_flag_count":   (0.70, 0.90),
        "flow_bytes_s":     (0.05, 0.15),
    },
    "web": {
        "fwd_bytes":        (0.80, 0.95),
        "bwd_bytes":        (0.20, 0.30),
        "psh_flag_count":   (0.85, 0.98),
        "fwd_psh_flags":    (0.85, 0.98),
        "flow_duration":    (0.05, 0.12),
        "flow_iat_mean":    (0.02, 0.08),
    },
    "custom": {
        # Generic high-variance anomaly: packet size, chaotic IAT, unusual ports
        "fwd_pkt_len_std":  (0.85, 0.99),
        "bwd_pkt_len_std":  (0.85, 0.99),
        "pkt_len_std":      (0.85, 0.99),
        "pkt_len_var":      (0.85, 0.99),
        "flow_iat_mean":    (0.02, 0.05),   # near-zero IAT
        "flow_iat_std":     (0.92, 0.99),   # chaotic
        "dest_port":        (0.90, 0.99),
    },
}
