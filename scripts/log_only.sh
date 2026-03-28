#!/usr/bin/env bash
# scripts/log_only.sh
# AURA Response — Tier 1: Passive Logging
# Called by policy_engine.py for LOW severity events.
# Environment variables set by caller:
#   NODE_ID       — synthetic node identifier (e.g. node_5)
#   SIMULATED_IP  — derived IP for audit trail (e.g. 10.0.0.5)
#   SEVERITY      — alert severity label
#   CONFIDENCE    — fused confidence score

echo "[AURA LOG_ONLY] $(date -u +%Y-%m-%dT%H:%M:%SZ) | node=${NODE_ID} ip=${SIMULATED_IP} severity=${SEVERITY} confidence=${CONFIDENCE}"
exit 0
