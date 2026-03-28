#!/usr/bin/env bash
# scripts/isolate.sh
# AURA Response — Tier 3: Full Network Isolation
# Called by policy_engine.py ONLY after HITL approval for HIGH + STANDARD severity.
# NEVER called for CRITICAL nodes (policy_engine enforces this via YAML rules).
# Environment variables set by caller:
#   NODE_ID       — synthetic node identifier (e.g. node_5)
#   SIMULATED_IP  — derived IP for iptables rule
#   SEVERITY      — alert severity label
#   CONFIDENCE    — fused confidence score

echo "[AURA ISOLATE] $(date -u +%Y-%m-%dT%H:%M:%SZ) | node=${NODE_ID} ip=${SIMULATED_IP} severity=${SEVERITY} confidence=${CONFIDENCE}"

# Drop all packets to/from the infected node
iptables -A INPUT  -s "${SIMULATED_IP}" -j DROP
iptables -A OUTPUT -d "${SIMULATED_IP}" -j DROP

echo "[AURA ISOLATE] Node ${NODE_ID} (${SIMULATED_IP}) removed from network map. Blast radius contained."
exit 0
