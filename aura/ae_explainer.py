"""
aura/ae_explainer.py — AE Feature Attribution & Attack Classification
=======================================================================

Given a per-feature reconstruction residual vector |x - x_hat| ∈ ℝ^78,
this module:

  1. Names the top contributing features in human-readable terms
  2. Matches the residual pattern against known attack signatures
  3. Produces a plain-English explanation panel for the SOC operator

Design
------
We use a lightweight dot-product similarity between the (normalized) residual
vector and pre-defined attack signature vectors.  Each signature encodes which
features SHOULD be anomalous for that attack type, weighted by expected severity.
No additional model is needed — it's a lookup/scoring pass on top of the AE.

This is interpretable-by-design: the operator can see exactly WHICH features
drove the alert AND why the system inferred a specific attack category.
"""

import numpy as np
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Feature Index → Human-Readable Name
# All 78 CICIDS2017 features (Label and IP columns stripped during preprocessing)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: Dict[int, str] = {
    0:  "Destination Port",
    1:  "Flow Duration",
    2:  "Total Fwd Packets",
    3:  "Total Bwd Packets",
    4:  "Total Fwd Bytes",
    5:  "Total Bwd Bytes",
    6:  "Fwd Pkt Len Max",
    7:  "Fwd Pkt Len Min",
    8:  "Fwd Pkt Len Mean",
    9:  "Fwd Pkt Len Std",
    10: "Bwd Pkt Len Max",
    11: "Bwd Pkt Len Min",
    12: "Bwd Pkt Len Mean",
    13: "Bwd Pkt Len Std",
    14: "Flow Bytes/s",
    15: "Flow Packets/s",
    16: "Flow IAT Mean",
    17: "Flow IAT Std",
    18: "Flow IAT Max",
    19: "Flow IAT Min",
    20: "Fwd IAT Total",
    21: "Fwd IAT Mean",
    22: "Fwd IAT Std",
    23: "Fwd IAT Max",
    24: "Fwd IAT Min",
    25: "Bwd IAT Total",
    26: "Bwd IAT Mean",
    27: "Bwd IAT Std",
    28: "Bwd IAT Max",
    29: "Bwd IAT Min",
    30: "Fwd PSH Flags",
    31: "Bwd PSH Flags",
    32: "Fwd URG Flags",
    33: "Bwd URG Flags",
    34: "Fwd Header Length",
    35: "Bwd Header Length",
    36: "Fwd Packets/s",
    37: "Bwd Packets/s",
    38: "Pkt Len Min",
    39: "Pkt Len Max",
    40: "Pkt Len Mean",
    41: "Pkt Len Std",
    42: "Pkt Len Var",
    43: "FIN Flag Count",
    44: "SYN Flag Count",
    45: "RST Flag Count",
    46: "PSH Flag Count",
    47: "ACK Flag Count",
    48: "URG Flag Count",
    49: "CWE Flag Count",
    50: "ECE Flag Count",
    51: "Down/Up Ratio",
    52: "Avg Pkt Size",
    53: "Avg Fwd Segment Size",
    54: "Avg Bwd Segment Size",
    55: "Fwd Avg Bytes/Bulk",
    56: "Fwd Avg Packets/Bulk",
    57: "Fwd Avg Bulk Rate",
    58: "Bwd Avg Bytes/Bulk",
    59: "Bwd Avg Packets/Bulk",
    60: "Bwd Avg Bulk Rate",
    61: "Subflow Fwd Packets",
    62: "Subflow Fwd Bytes",
    63: "Subflow Fwd Bytes (2)",
    64: "Subflow Bwd Packets",
    65: "Subflow Bwd Bytes",
    66: "Init Win Bytes Fwd",
    67: "Init Win Bytes Bwd",
    68: "Act Data Pkt Fwd",
    69: "Min Seg Size Fwd",
    70: "Active Mean",
    71: "Active Std",
    72: "Active Max",
    73: "Active Min",
    74: "Idle Mean",
    75: "Idle Std",
    76: "Idle Max",
    77: "Idle Min",
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature Groups (for grouped explanation display)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_GROUPS: Dict[str, List[int]] = {
    "Volume / Bytes":   [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 62, 63, 64, 65],
    "Bandwidth Rates":  [14, 15, 36, 37, 51, 52, 53, 54],
    "Timing / IAT":     [1, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    "TCP Flags":        [30, 31, 32, 33, 43, 44, 45, 46, 47, 48, 49, 50],
    "Idle / Active":    [70, 71, 72, 73, 74, 75, 76, 77],
    "Bulk Transfer":    [55, 56, 57, 58, 59, 60],
    "Window / Segment": [66, 67, 68, 69],
}


# ─────────────────────────────────────────────────────────────────────────────
# Attack Signature Vectors
# ─────────────────────────────────────────────────────────────────────────────
# Each signature is a sparse dict: {feature_index: expected_high_residual_weight}
# Values are relative importances — they get L2-normalised at match time.
# Based on the CICIDS2017 attack taxonomy + attack_injector.py profiles.

ATTACK_SIGNATURES: Dict[str, Dict[int, float]] = {
    "DDoS": {
        15: 1.0,   # Flow Packets/s
        14: 0.9,   # Flow Bytes/s
        44: 0.8,   # SYN Flag Count
        2:  0.8,   # Total Fwd Packets
        16: 0.7,   # Flow IAT Mean
        17: 0.7,   # Flow IAT Std
    },
    "Port Scan": {
        45: 1.0,   # RST Flag Count
        44: 0.9,   # SYN Flag Count
        1:  0.8,   # Flow Duration
        4:  0.7,   # Total Fwd Bytes
        5:  0.7,   # Total Bwd Bytes
        14: 0.5,   # Flow Bytes/s
    },
    "Lateral Movement": {
        17: 1.0,   # Flow IAT Std
        74: 0.9,   # Idle Mean
        1:  0.8,   # Flow Duration
        2:  0.7,   # Total Fwd Packets
        75: 0.6,   # Idle Std
        46: 0.5,   # PSH Flag Count
    },
    "Data Exfiltration": {
        4:  1.0,   # Total Fwd Bytes
        63: 0.9,   # Subflow Fwd Bytes
        1:  0.8,   # Flow Duration
        16: 0.7,   # Flow IAT Mean
        17: 0.6,   # Flow IAT Std
        5:  0.5,   # Total Bwd Bytes
        65: 0.5,   # Subflow Bwd Bytes
    },
    "Web Attack": {
        46: 1.0,   # PSH Flag Count
        4:  0.9,   # Total Fwd Bytes
        5:  0.8,   # Total Bwd Bytes
        1:  0.7,   # Flow Duration
        16: 0.6,   # Flow IAT Mean
    },
}


# Human-readable explanations per attack type
ATTACK_EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "DDoS": {
        "icon":    "🌊",
        "summary": "Volumetric flood attack detected",
        "detail":  (
            "Packet rate and bandwidth are abnormally high with near-zero "
            "inter-arrival time — consistent with a UDP/SYN flood. "
            "Incomplete TCP handshakes (high SYN, low ACK) confirm the source "
            "is NOT establishing legitimate connections. "
            "Action: rate-limit the source subnet and engage upstream scrubbing."
        ),
        "why_high": "Flow Packets/s and SYN flags are the primary drivers — "
                    "the model has never seen legitimate traffic at this rate.",
    },
    "Port Scan": {
        "icon":    "🔍",
        "summary": "Network reconnaissance / port scan detected",
        "detail":  (
            "Multiple extremely short flows with minimal byte transfer and "
            "high RST + SYN flag counts — the attacker is probing which services "
            "are open without completing any connection. "
            "Action: block the scanning IP, alert vulnerability management team."
        ),
        "why_high": "RST Flag Count and very short Flow Duration are the primary "
                    "drivers — legitimate flows do not terminate this abruptly en masse.",
    },
    "Lateral Movement": {
        "icon":    "↔️",
        "summary": "Internal lateral movement / east-west threat detected",
        "detail":  (
            "High timing jitter (Flow IAT Std) combined with long idle periods "
            "between bursts is the hallmark of a compromised host performing "
            "internal reconnaissance. The GNN (Layer 2) should confirm abnormal "
            "device-to-device connectivity not seen during training. "
            "Action: isolate the source host, initiate EDR investigation."
        ),
        "why_high": "Flow IAT Std and Idle Mean are the primary drivers — "
                    "the beacon-like sleep-burst pattern is not present in normal flows.",
    },
    "Data Exfiltration": {
        "icon":    "📤",
        "summary": "Data exfiltration (low & slow) detected",
        "detail":  (
            "Extreme asymmetry: large forward (outbound) byte count vs near-zero "
            "backward (inbound) bytes over a sustained, long connection. "
            "Robotic inter-arrival timing (low Std) indicates machine-scripted "
            "exfiltration rather than human-driven traffic. "
            "Action: terminate the connection, inspect endpoint for malware, "
            "check DLP logs for data classification hits."
        ),
        "why_high": "Total Fwd/Bwd Bytes ratio and Subflow Fwd Bytes are the primary "
                    "drivers — upload-only sustained flows are outside the normal manifold.",
    },
    "Web Attack": {
        "icon":    "💉",
        "summary": "Web application attack detected (SQLi / XSS)",
        "detail":  (
            "Elevated PSH flags and large forward payload sizes on short-duration "
            "flows suggest HTTP request manipulation — consistent with SQL injection "
            "or XSS payloads being submitted. "
            "Action: review WAF logs, block the offending IP, audit database "
            "query logs for injection attempts."
        ),
        "why_high": "Fwd PSH Flags and PSH Flag Count are primary drivers — "
                    "legitimate HTTP traffic does not push this many payloads per flow.",
    },
    "Unknown Anomaly": {
        "icon":    "❓",
        "summary": "Anomalous pattern — no close attack signature match",
        "detail":  (
            "The reconstruction error is elevated but the feature residual pattern "
            "does not closely match any known attack signature. This may indicate "
            "a novel attack variant, misconfigured device, or legitimate but unusual "
            "traffic pattern. "
            "Action: review the top contributing features manually and escalate "
            "to Tier-2 analysis."
        ),
        "why_high": "Spread residuals across multiple unrelated feature groups — "
                    "no single attack taxonomy matches well.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Core Explanation Function
# ─────────────────────────────────────────────────────────────────────────────

def explain_ae(
    residuals:  np.ndarray,   # [78] mean absolute per-feature residual
    top_k:      int = 5,
    min_score:  float = 0.05, # minimum similarity to claim a match
) -> Dict:
    """
    Given a per-feature reconstruction residual vector, return a structured
    explanation dict for the dashboard.

    Parameters
    ----------
    residuals  : np.ndarray [78] — mean |x - x_hat| per feature
    top_k      : how many top features to surface
    min_score  : cosine similarity threshold below which we say "Unknown"

    Returns
    -------
    dict with keys:
      top_features    : list of (feature_name, residual_value, feature_index)
      group_residuals : dict {group_name: mean_residual}
      inferred_attack : str — matched attack label (or "Unknown Anomaly")
      match_score     : float ∈ [0,1]
      explanation     : dict (icon, summary, detail, why_high)
    """
    residuals = np.array(residuals, dtype=np.float32)
    n_feats   = len(residuals)

    # ── Top-K contributing features ───────────────────────────────────────
    top_indices = np.argsort(residuals)[::-1][:top_k]
    top_features = [
        (FEATURE_NAMES.get(int(i), f"Feature_{i}"), float(residuals[i]), int(i))
        for i in top_indices
    ]

    # ── Group-level residuals ─────────────────────────────────────────────
    group_residuals: Dict[str, float] = {}
    for group_name, indices in FEATURE_GROUPS.items():
        valid = [residuals[i] for i in indices if i < n_feats]
        group_residuals[group_name] = float(np.mean(valid)) if valid else 0.0

    # ── Attack signature matching (cosine similarity) ─────────────────────
    # Build a dense residual vector from the sparse signature
    best_attack = "Unknown Anomaly"
    best_score  = 0.0

    r_norm = np.linalg.norm(residuals)
    if r_norm > 1e-8:
        r_unit = residuals / r_norm

        for attack_name, sig_dict in ATTACK_SIGNATURES.items():
            # Build dense signature vector
            sig_vec = np.zeros(n_feats, dtype=np.float32)
            for feat_idx, weight in sig_dict.items():
                if feat_idx < n_feats:
                    sig_vec[feat_idx] = weight

            sig_norm = np.linalg.norm(sig_vec)
            if sig_norm < 1e-8:
                continue

            sig_unit = sig_vec / sig_norm
            score    = float(np.dot(r_unit, sig_unit))   # cosine similarity

            if score > best_score:
                best_score  = score
                best_attack = attack_name

    if best_score < min_score:
        best_attack = "Unknown Anomaly"

    explanation = ATTACK_EXPLANATIONS.get(best_attack, ATTACK_EXPLANATIONS["Unknown Anomaly"])

    return {
        "top_features":    top_features,
        "group_residuals": group_residuals,
        "inferred_attack": best_attack,
        "match_score":     round(best_score, 3),
        "explanation":     explanation,
    }
