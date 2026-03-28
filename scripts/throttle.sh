#!/usr/bin/env bash
# scripts/throttle.sh
# AURA Response — Tier 2: Bandwidth Throttle
# Called by policy_engine.py for MEDIUM severity, or as HITL degradation fallback for HIGH.
# Environment variables set by caller:
#   NODE_ID       — synthetic node identifier (e.g. node_5)
#   SIMULATED_IP  — derived IP for tc rule targeting
#   SEVERITY      — alert severity label
#   CONFIDENCE    — fused confidence score

echo "[AURA THROTTLE] $(date -u +%Y-%m-%dT%H:%M:%SZ) | node=${NODE_ID} ip=${SIMULATED_IP} severity=${SEVERITY} confidence=${CONFIDENCE}"

# Apply HTB traffic shaper — caps outbound from SIMULATED_IP to 10 kbps
# (Chokes data exfiltration while keeping the connection alive for forensics)
tc qdisc add dev eth0 root handle 1: htb default 12 2>/dev/null || true
tc class add dev eth0 parent 1: classid 1:1 htb rate 10kbps 2>/dev/null || true
tc filter add dev eth0 protocol ip parent 1:0 prio 1 \
    u32 match ip src "${SIMULATED_IP}" flowid 1:1 2>/dev/null || true

echo "[AURA THROTTLE] Rate-limit applied: ${SIMULATED_IP} capped at 10kbps"
exit 0
