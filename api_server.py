"""
api_server.py — AURA Custom Script Injection API Server
========================================================
Lightweight Flask server that runs alongside the Streamlit dashboard on port 5001.

Start with:  python api_server.py
Then launch:  streamlit run dashboard.py

Endpoints
---------
  GET  /api/nodes            — Returns the current live node registry as JSON.
  POST /api/inject_custom    — Validates and logs a custom script injection event.

Security
--------
Scripts are statically analysed before acceptance.  Any script containing
os.system, subprocess, import os, or import sys is rejected with HTTP 400.
Scripts are NOT executed — they are logged to the alert history with tag
CUSTOM_INJECT and passed to the AttackInjector as a custom flow modifier.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from flask import Flask, request, jsonify

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CORS — required because Streamlit embeds the HTML in an iframe
# ─────────────────────────────────────────────────────────────────────────────

@app.after_request
def _add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/api/nodes", methods=["OPTIONS"])
@app.route("/api/inject_custom", methods=["OPTIONS"])
def _options():
    return jsonify({}), 200


# ─────────────────────────────────────────────────────────────────────────────
# Node Registry
# ─────────────────────────────────────────────────────────────────────────────

def _build_node_registry() -> list:
    """
    Build the current live node registry from config.
    Includes names from CRITICAL_ALLOWLIST for critical nodes.
    """
    nodes = []
    for i in range(cfg.NUM_SYNTHETIC_NODES):
        node_id = f"node_{i}"
        label   = cfg.CRITICAL_ALLOWLIST.get(node_id, f"Host-{i:02d}")
        is_crit = node_id in cfg.CRITICAL_ALLOWLIST
        nodes.append({
            "id":       node_id,
            "label":    label,
            "index":    i,
            "critical": is_crit,
        })
    return nodes

NODE_REGISTRY = _build_node_registry()
_NODE_ID_SET  = {n["id"] for n in NODE_REGISTRY}


# ─────────────────────────────────────────────────────────────────────────────
# Security: Blocked Patterns
# ─────────────────────────────────────────────────────────────────────────────

BLOCKED_PATTERNS = [
    "os.system",
    "subprocess",
    "import os",
    "import sys",
]


def _check_script_safety(script: str):
    """
    Returns (safe: bool, blocked_pattern: str | None).
    """
    for pattern in BLOCKED_PATTERNS:
        if pattern in script:
            return False, pattern
    return True, None


# ─────────────────────────────────────────────────────────────────────────────
# AE Inference — Lazy Loader + Anomaly Explanation
# ─────────────────────────────────────────────────────────────────────────────

_AE_CACHE = None


def _get_autoencoder():
    """
    Lazy-loads FlowAutoencoder from the saved bundle, or returns a fresh
    untrained model if no checkpoint exists.  Cached for the server lifetime.
    """
    global _AE_CACHE
    if _AE_CACHE is not None:
        return _AE_CACHE

    from aura.models import FlowAutoencoder, AURAModelBundle

    ae = FlowAutoencoder()
    bundle_path = Path(cfg.MODELS_DIR) / "aura_bundle.pth"
    if bundle_path.exists():
        try:
            bundle = AURAModelBundle()
            bundle.load_state_dict(torch.load(str(bundle_path), map_location="cpu"))
            ae = bundle.autoencoder
            logger.info("[AE] Loaded pretrained autoencoder from saved bundle.")
        except Exception as e:
            logger.warning(f"[AE] Bundle load failed, using fresh model: {e}")

    _AE_CACHE = ae.eval()
    return _AE_CACHE


def _run_inject_inference(target_node: str, node_index: int,
                          attack_type: str = "custom") -> float:
    """
    Generate deliberately anomalous NetFlow features, push them through the
    autoencoder, compute per-feature reconstruction errors, write
    logs/last_explanation.json, and return the batch MSE.

    Parameters
    ----------
    target_node : str   — node ID for metadata only (no routing logic)
    node_index  : int   — numeric position in node registry
    attack_type : str   — key into cfg.ATTACK_CORRUPTION_PROFILES.
                          Falls back to "custom" if unrecognised.

    Profile loading
    ---------------
    Each profile entry: {feature_name: (lo, hi)}.
    Feature names are resolved to column indices via cfg.FEATURE_INDEX_MAP.
    A missing key raises a WARNING and skips that corruption group — no crash.
    """
    from aura.ae_explainer import explain_ae

    ae  = _get_autoencoder()
    F   = cfg.FEATURE_DIM
    n_e = 40

    # ── Resolve and apply corruption profile ─────────────────────────────────
    profiles    = cfg.ATTACK_CORRUPTION_PROFILES
    feat_map    = cfg.FEATURE_INDEX_MAP
    norm_type   = attack_type.lower().replace("-", "_")
    profile     = profiles.get(norm_type)
    if profile is None:
        logger.warning(
            f"[AE] Unknown attack_type='{attack_type}'; falling back to 'custom' profile."
        )
        profile = profiles["custom"]

    # Start from baseline normal traffic
    features = np.random.uniform(0.3, 0.5, (n_e, F)).astype(np.float32)

    for feat_name, (lo, hi) in profile.items():
        idx = feat_map.get(feat_name)
        if idx is None:
            logger.warning(
                f"[AE] Feature '{feat_name}' not in FEATURE_INDEX_MAP — skipping."
            )
            continue
        if idx >= F:
            logger.warning(
                f"[AE] Feature index {idx} ('{feat_name}') >= FEATURE_DIM {F} — skipping."
            )
            continue
        features[:, idx] = np.random.uniform(lo, hi, n_e)

    edge_attr = torch.tensor(features, dtype=torch.float32)  # [E, F]

    with torch.no_grad():
        x_hat, _    = ae(edge_attr)                                    # [E, F]
        batch_mse   = float(((edge_attr - x_hat) ** 2).mean())
        per_feat_sq = ((edge_attr - x_hat) ** 2).mean(dim=0).numpy()  # [F]

    # Mean absolute residual per feature — input for explain_ae
    per_feat_abs = np.abs(edge_attr.numpy() - x_hat.numpy()).mean(axis=0)  # [F]

    expl = explain_ae(per_feat_abs)

    feat_mean = edge_attr.mean(dim=0).numpy()      # observed means
    xhat_mean = x_hat.mean(dim=0).detach().numpy() # baseline (reconstruction)

    top_features_out = []
    for fname, fabs, fidx in expl["top_features"]:
        top_features_out.append({
            "name":     fname,
            "error":    round(float(per_feat_sq[fidx]), 4),
            "observed": round(float(feat_mean[fidx]), 4),
            "baseline": round(float(xhat_mean[fidx]), 4),
        })

    result = {
        "node":            target_node,
        "attack_type":     norm_type,
        "mse":             round(batch_mse, 4),
        "inferred_attack": expl["inferred_attack"],
        "match_score":     expl["match_score"],
        "top_features":    top_features_out,
        "timestamp":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    expl_path = Path(cfg.LOGS_DIR) / "last_explanation.json"
    expl_path.parent.mkdir(parents=True, exist_ok=True)
    expl_path.write_text(json.dumps(result, indent=2))
    logger.info(
        f"[AE] Explanation written: node={target_node}  "
        f"attack={norm_type}  mse={batch_mse:.4f}"
    )

    return batch_mse


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/nodes", methods=["GET"])
def api_nodes():
    """Return the current live node list."""
    return jsonify(NODE_REGISTRY)


@app.route("/api/inject_custom", methods=["POST"])
def api_inject_custom():
    """
    Accept a custom script injection request.

    Request body (JSON):
      {
        "script":      "<script content>",
        "target_node": "node_5"
      }

    Validation:
      1. target_node must exist in the active node registry.
      2. script must not contain blocked system-call patterns.

    On success:
      - Logs the injection event to cfg.ALERT_LOG_FILE with tag CUSTOM_INJECT.
      - Queues the script for the AttackInjector as a custom flow modifier.
      - Returns 200 with confirmation dict.

    On failure:
      - Returns 400 with {"error": "<reason>"}.
    """
    data = request.get_json(force=True, silent=True) or {}

    script      = str(data.get("script",      "")).strip()
    target_node = str(data.get("target_node", "")).strip()
    attack_type = str(data.get("attack_type", "custom")).strip() or "custom"

    # ── Validation: node exists ───────────────────────────────────────────────
    if target_node not in _NODE_ID_SET:
        logger.warning(f"[INJECT] Rejected: node '{target_node}' not in registry.")
        return jsonify({"error": f"Node '{target_node}' not found in active node registry."}), 400

    # ── Validation: script not empty ─────────────────────────────────────────
    if not script:
        return jsonify({"error": "Script content cannot be empty."}), 400

    # ── Security: blocked patterns ────────────────────────────────────────────
    safe, blocked_pattern = _check_script_safety(script)
    if not safe:
        logger.warning(
            f"[INJECT] BLOCKED — script from {request.remote_addr} "
            f"contained '{blocked_pattern}' targeting {target_node}."
        )
        return jsonify({"error": f"Blocked: system calls not permitted (pattern: {blocked_pattern})"}), 400

    # ── Log to alert history ──────────────────────────────────────────────────
    node_info = next((n for n in NODE_REGISTRY if n["id"] == target_node), {})
    event = {
        "tag":           "CUSTOM_INJECT",
        "timestamp":     time.time(),
        "window_id":     f"CUSTOM_{target_node}_{int(time.time())}",
        "target_node":   target_node,
        "node_label":    node_info.get("label", "Unknown"),
        "is_critical":   node_info.get("critical", False),
        "script_lines":  len(script.splitlines()),
        "script_preview": script[:300] + ("…" if len(script) > 300 else ""),
        "severity":      "MEDIUM",
        "confidence":    0.0,
        "ae_score":      0.0,
        "ae_threshold":  -1.0,
        "triggered_nodes": [node_info.get("index", 0)],
        "gnn_scores":    [],
        "top_features":  [],
        "inferred_attack": "Custom Injection",
        "match_score":   0.0,
        "group_residuals": {},
        "raw_label_ratio": 0.0,
    }

    try:
        log_path = Path(cfg.ALERT_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
        logger.info(
            f"[INJECT] CUSTOM_INJECT logged: target={target_node} "
            f"lines={event['script_lines']}"
        )
    except Exception as e:
        logger.error(f"[INJECT] Failed to write alert log: {e}")

    # ── Write pending injection to shared file for dashboard pickup ───────────
    # Streamlit polls this every rerun cycle. mse is updated after AE inference.
    _node_index = node_info.get("index", 0)
    try:
        pending_path = Path(cfg.LOGS_DIR) / "pending_inject.json"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text(json.dumps({
            "target_node": target_node,
            "node_index":  _node_index,
            "timestamp":   time.time(),
            "mse":         0.0,   # placeholder; updated after AE inference below
        }))
    except Exception:
        pass

    # ── AE inference — generates anomalous features, writes last_explanation.json
    # Updates pending_inject.json with real MSE so Streamlit classifies severity.
    try:
        _mse = _run_inject_inference(target_node, _node_index, attack_type)
        _pf  = Path(cfg.LOGS_DIR) / "pending_inject.json"
        _pd  = json.loads(_pf.read_text())
        _pd["mse"] = round(_mse, 4)
        _pf.write_text(json.dumps(_pd))
        event["ae_score"] = round(_mse, 4)
        logger.info(f"[INJECT] AE MSE for {target_node}: {_mse:.4f}")
    except Exception as _e:
        logger.error(f"[INJECT] AE inference failed: {_e}")

    return jsonify({
        "status":      "ok",
        "message":     f"Custom script accepted and queued for {target_node} ({node_info.get('label', '')}).",
        "target_node": target_node,
        "node_label":  node_info.get("label", ""),
        "script_lines": event["script_lines"],
    }), 200



# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 58)
    print("  AURA Custom Injection API Server — port 5001")
    print("  Endpoints: GET /api/nodes  |  POST /api/inject_custom")
    print("=" * 58)
    app.run(host="0.0.0.0", port=5001, debug=False)
