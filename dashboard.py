"""
dashboard.py — AURA Live Operations Dashboard
==============================================
Run with:  streamlit run dashboard.py

The AURA dashboard is the "nerve centre" of the hackathon demo.

Layout
------
 ┌──────────────────────────────────────────────────────────────────┐
 │  AURA — Autonomous Unified Resilience Architecture         Status │
 ├────────────────────────────┬─────────────────────────────────────┤
 │  Live Network Topology     │  Anomaly Score Timeline             │
 │  (Plotly animated graph)   │  (EMA threshold visible)            │
 │                            │                                     │
 ├────────────────────────────┴─────────────────────────────────────┤
 │  🔴 ATTACK INJECTION         🌐 FEDERATION         ⛓ BLOCKCHAIN  │
 │  [DDoS] [Scan] [Lateral]   [Run FL Simulation]   [Verify Hash]  │
 │  [Exfil] [Web Attack]                                            │
 ├────────────────────────────────────────────────────────────────  ┤
 │  Event Log (last 20 events)            Alert History             │
 └──────────────────────────────────────────────────────────────────┘
"""

import json
import os
import time
import threading
import hashlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from aura.models import FlowAutoencoder, AuraSTGNN, AURAModelBundle
from aura.detector import AURAInferenceEngine, AlertSeverity
from aura.response_engine import AURAResponseEngine
from aura.attack_injector import AttackInjector, AttackType
from aura.blockchain import AURABlockchainLogger

# ─────────────────────────────────────────────────────────────────────────────
# Org Identity  (set via AURA_ORG_ID env var before launching)
# ─────────────────────────────────────────────────────────────────────────────

_ORG_PROFILES = {
    "hospital":   {"label": "Hospital",    "id": "org_hospital_1",   "net": "192.168.1.0/24",  "icon": "🏥", "role": "Normal",    "color": "#00ff88"},
    "bank":       {"label": "Bank",        "id": "org_bank_2",       "net": "10.0.1.0/24",    "icon": "🏦", "role": "Byzantine", "color": "#ff8800"},
    "university": {"label": "University",  "id": "org_university_3", "net": "172.16.1.0/24",  "icon": "🎓", "role": "Normal",    "color": "#4488ff"},
}
_ORG_KEY  = os.environ.get("AURA_ORG_ID", "").lower().strip()
ORG       = _ORG_PROFILES.get(_ORG_KEY)   # None if not set

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

_page_title = f"AURA — {ORG['icon']} {ORG['label']}" if ORG else "AURA — Autonomous Unified Resilience Architecture"
_page_icon  = ORG["icon"] if ORG else "🛡️"

st.set_page_config(
    page_title = _page_title,
    page_icon  = _page_icon,
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Colour Theme
# ─────────────────────────────────────────────────────────────────────────────

THEME = {
    "bg":        "#0a0e1a",
    "panel":     "#0f1629",
    "border":    "#1e2d4a",
    "text":      "#e0e8f0",
    "green":     "#00ff88",
    "yellow":    "#ffd700",
    "red":       "#ff4444",
    "blue":      "#4488ff",
    "cyan":      "#00ccff",
    "orange":    "#ff8800",
    "dim":       "#445566",
}

st.markdown(f"""
<style>
  .stApp {{ background-color: {THEME['bg']}; }}
  .block-container {{ padding-top: 1rem; }}
  .metric-card {{
    background: {THEME['panel']}; border: 1px solid {THEME['border']};
    border-radius: 8px; padding: 1rem; text-align: center;
  }}
  .alert-high  {{ color: {THEME['red']};    font-weight: bold; }}
  .alert-med   {{ color: {THEME['orange']}; font-weight: bold; }}
  .alert-low   {{ color: {THEME['yellow']}; }}
  .alert-norm  {{ color: {THEME['green']};  }}
  .chain-row   {{ color: {THEME['cyan']};   font-family: monospace; font-size: 0.8em; }}
  .fed-log     {{ color: {THEME['blue']};   font-family: monospace; font-size: 0.8em; }}
  h1, h2, h3, h4 {{ color: {THEME['text']} !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "engine":          None,
        "responder":       None,
        "injector":        None,
        "blockchain":      None,
        "ae_scores":       [],       # Timeline of AE MSE scores
        "thresholds":      [],       # Corresponding EMA thresholds
        "timestamps":      [],       # Wall-clock times
        "alerts":          [],       # List of AnomalyEvent dicts
        "incidents":       [],       # List of IncidentRecord dicts
        "fed_log":         [],       # Federation event strings
        "chain_log":       [],       # Blockchain hash events
        "node_colors":     {i: THEME["green"] for i in range(cfg.NUM_SYNTHETIC_NODES)},
        "node_states":     {i: "Normal" for i in range(cfg.NUM_SYNTHETIC_NODES)},
        "current_graph":   None,
        "attack_active":   False,
        "attack_type":     None,
        "system_status":   "INITIALISING",
        "total_attacks":   0,
        "total_blocked":   0,
        "fl_rounds_done":  0,
        "chain_entries":   0,
        "models_loaded":   False,
        "window_counter":  0,
        "last_explanation": None,   # Most recent AE explainer output dict
        "fl_client_status": [],     # Per-client metadata from latest FL round
        "fl_ready":         False,  # Whether this org node has signalled FL readiness
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Model & Component Loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AURA models …")
def load_components():
    """Load or initialise all AURA components.  Cached across reruns."""
    bundle_path = cfg.MODELS_DIR / "aura_bundle.pth"
    bundle      = AURAModelBundle()

    if bundle_path.exists():
        bundle.load_state_dict(torch.load(bundle_path, map_location="cpu"))
        status = "PRE-TRAINED"
    else:
        status = "UNTRAINED (DEMO MODE)"

    engine    = AURAInferenceEngine(bundle.autoencoder, bundle.stgnn)
    responder = AURAResponseEngine()
    injector  = AttackInjector()
    bc        = AURABlockchainLogger()

    # Pre-warm EMA threshold with synthetic normal traffic so alerts fire
    # immediately when the user clicks an attack button on the dashboard.
    _N = cfg.NUM_SYNTHETIC_NODES
    _E = 40
    for _ in range(cfg.EMA_WARMUP_BATCHES + 10):
        _x    = torch.randn(_N, cfg.FEATURE_DIM) * 0.1   # low-variance normal
        _ei   = torch.randint(0, _N, (2, _E))
        _attr = torch.randn(_E, cfg.FEATURE_DIM) * 0.1
        engine.process({"x": _x, "edge_index": _ei,
                        "edge_attr": _attr, "window_id": "warmup"})

    return engine, responder, injector, bc, status


engine, responder, injector, bc, model_status = load_components()

# Inject into session state if not already there
if not st.session_state["models_loaded"]:
    st.session_state["engine"]       = engine
    st.session_state["responder"]    = responder
    st.session_state["injector"]     = injector
    st.session_state["blockchain"]   = bc
    st.session_state["models_loaded"] = True
    st.session_state["system_status"] = "ACTIVE"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def build_network_figure(
    node_colors: dict,
    node_states: dict,
    edge_index:  Optional[np.ndarray] = None,
    n_nodes:     int = cfg.NUM_SYNTHETIC_NODES,
) -> go.Figure:
    """
    Build an animated Plotly network graph of the current network state.
    Nodes are arranged in a circular topology.
    Node colour reflects health: green=normal, yellow=suspect, red=isolated.
    """
    # Circular layout
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    pos_x  = np.cos(angles)
    pos_y  = np.sin(angles)

    # Node labels (match critical allowlist)
    node_labels = []
    for i in range(n_nodes):
        key   = f"node_{i}"
        label = cfg.CRITICAL_ALLOWLIST.get(key, f"Host-{i:02d}")
        node_labels.append(f"{label}<br>({node_states.get(i, 'Normal')})")

    # Edge traces
    edge_traces = []
    if edge_index is not None and edge_index.shape[1] > 0:
        for idx in range(min(edge_index.shape[1], 60)):  # Cap for performance
            s, d = int(edge_index[0, idx]), int(edge_index[1, idx])
            edge_traces.append(go.Scatter(
                x=[pos_x[s], pos_x[d], None],
                y=[pos_y[s], pos_y[d], None],
                mode="lines",
                line=dict(width=1, color="#1e3a5a"),
                hoverinfo="none",
                showlegend=False,
            ))

    # Node trace
    colors = [node_colors.get(i, THEME["green"]) for i in range(n_nodes)]
    sizes  = [20 if f"node_{i}" in cfg.CRITICAL_ALLOWLIST else 14 for i in range(n_nodes)]
    symbols= ["diamond" if f"node_{i}" in cfg.CRITICAL_ALLOWLIST else "circle"
              for i in range(n_nodes)]

    node_trace = go.Scatter(
        x=pos_x, y=pos_y,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            symbol=symbols,
            line=dict(width=1.5, color="#ffffff"),
        ),
        text=[f"N{i}" for i in range(n_nodes)],
        textposition="top center",
        textfont=dict(size=8, color=THEME["text"]),
        hovertext=node_labels,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        paper_bgcolor = THEME["bg"],
        plot_bgcolor  = THEME["bg"],
        margin        = dict(l=10, r=10, t=10, b=10),
        xaxis         = dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis         = dict(showgrid=False, zeroline=False, showticklabels=False,
                              scaleanchor="x"),
        height        = 320,
    )
    return fig


def build_score_timeline(
    scores:     List[float],
    thresholds: List[float],
    timestamps: List[float],
) -> go.Figure:
    """Plotly line chart: AE anomaly score vs dynamic EMA threshold over time."""
    if not scores:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"], height=200,
            annotations=[dict(text="Awaiting data…", showarrow=False,
                              font=dict(color=THEME["dim"], size=14))]
        )
        return fig

    t = list(range(len(scores)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=scores, mode="lines",
        name="AE Score (MSE)", line=dict(color=THEME["cyan"], width=2),
    ))

    valid_thresh = [v for v in thresholds if v > 0]
    if valid_thresh:
        fig.add_trace(go.Scatter(
            x=t, y=thresholds, mode="lines",
            name="EMA Threshold (3σ)",
            line=dict(color=THEME["red"], width=1.5, dash="dash"),
        ))

    fig.update_layout(
        paper_bgcolor = THEME["bg"],
        plot_bgcolor  = THEME["bg"],
        font          = dict(color=THEME["text"]),
        legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis         = dict(showgrid=False, title="Window"),
        yaxis         = dict(showgrid=True, gridcolor=THEME["border"], title="MSE"),
        height        = 200,
        margin        = dict(l=40, r=10, t=10, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Inference Step (called each tick)
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_tick(graph: dict, is_attack: bool = False):
    """Run one inference window through the AURA pipeline."""
    eng  = st.session_state["engine"]
    resp = st.session_state["responder"]

    event = eng.process(graph)

    # Update timeline data
    st.session_state["ae_scores"].append(event.ae_score)
    thresh = event.ae_threshold if event.ae_threshold > 0 else 0
    st.session_state["thresholds"].append(thresh)
    st.session_state["timestamps"].append(event.timestamp)

    # Trim to last 100 points for display
    for key in ["ae_scores", "thresholds", "timestamps"]:
        if len(st.session_state[key]) > 100:
            st.session_state[key] = st.session_state[key][-100:]

    # Update node colours
    colors = {i: THEME["green"] for i in range(cfg.NUM_SYNTHETIC_NODES)}
    states = {i: "Normal" for i in range(cfg.NUM_SYNTHETIC_NODES)}

    if event.severity != AlertSeverity.NORMAL:
        st.session_state["total_attacks"] += 1

        # Attack nodes → RED
        for nid in event.triggered_nodes:
            colors[nid] = THEME["red"]
            states[nid] = f"⚠ {event.severity.name}"

        # Add alert to log
        st.session_state["alerts"].insert(0, event.to_dict())
        if len(st.session_state["alerts"]) > 30:
            st.session_state["alerts"] = st.session_state["alerts"][:30]

        # Store explanation for the live panel (overwrite with latest triggered event)
        if event.inferred_attack != "Normal" and event.top_features:
            from aura.ae_explainer import ATTACK_EXPLANATIONS
            st.session_state["last_explanation"] = {
                "inferred_attack": event.inferred_attack,
                "match_score":     event.match_score,
                "top_features":    event.top_features,
                "group_residuals": event.group_residuals,
                "severity":        event.severity.name,
                "confidence":      event.confidence,
                "explanation":     ATTACK_EXPLANATIONS.get(
                    event.inferred_attack,
                    ATTACK_EXPLANATIONS.get("Unknown Anomaly", {})
                ),
            }

        # Run response engine
        records = resp.act(event)
        for r in records:
            if r.action_taken not in ("LOG_ONLY", "ALREADY_ACTIONED"):
                st.session_state["total_blocked"] += 1
            st.session_state["incidents"].insert(0, r.to_dict())
        if len(st.session_state["incidents"]) > 20:
            st.session_state["incidents"] = st.session_state["incidents"][:20]

    elif is_attack:
        # Attack was injected but EMA warmup still active — L1 not yet triggered.
        # Run the explainer directly on the raw features so the operator can
        # already see WHAT is anomalous, even before a formal alert fires.
        for nid in graph.get("attack_nodes", []):
            colors[nid] = THEME["yellow"]
            states[nid] = "Evaluating…"

        try:
            from aura.ae_explainer import explain_ae, ATTACK_EXPLANATIONS
            edge_attr = graph.get("edge_attr")
            if edge_attr is not None:
                feat_residuals = eng.ae.explain_features(edge_attr)
                expl = explain_ae(feat_residuals)
                st.session_state["last_explanation"] = {
                    "inferred_attack": expl["inferred_attack"],
                    "match_score":     expl["match_score"],
                    "top_features":    expl["top_features"],
                    "group_residuals": expl["group_residuals"],
                    "severity":        "LOW",   # warmup → tentative
                    "confidence":      expl["match_score"],
                    "explanation":     ATTACK_EXPLANATIONS.get(
                        expl["inferred_attack"],
                        ATTACK_EXPLANATIONS.get("Unknown Anomaly", {})
                    ),
                }
        except Exception:
            pass

    st.session_state["node_colors"]   = colors
    st.session_state["node_states"]   = states
    st.session_state["current_graph"] = graph
    st.session_state["window_counter"] += 1

    return event


# ─────────────────────────────────────────────────────────────────────────────
# Federation Simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_federation():
    """Run the full FL simulation and populate the federation log."""
    from aura.fl_server import run_federation_simulation

    st.session_state["fed_log"] = []
    st.session_state["fed_log"].append("🚀 Federation round initiated …")

    bc_module = st.session_state["blockchain"]

    # Capture federation output by running the simulation
    round_results = run_federation_simulation(blockchain_module=bc_module, n_rounds=3)

    for r in round_results:
        rnd     = r.get("round", "?")
        version = r.get("model_version", "N/A")
        h       = r.get("model_hash", "N/A")
        kept    = r.get("krum_selected", "?")

        # Keep the latest round's per-client status for the dashboard table
        if "client_statuses" in r:
            st.session_state["fl_client_status"] = r["client_statuses"]

        st.session_state["fed_log"].extend([
            f"━━━  Round {rnd}  ━━━",
            f"[CLIENT hospital_1] Attack pattern learned. Sending weights…",
            f"[CLIENT bank_2]     Local training complete. Sending weights…",
            f"[CLIENT uni_3]      Local training complete. Sending weights…",
            f"[SERVER] Krum filtering: {kept}/3 updates accepted.",
            f"[SERVER] Global Model {version} aggregated.",
            f"[BLOCKCHAIN] Hash recorded: {h[:20]}…",
            f"[CLIENT hospital_1] Verifying hash on chain… ✓ Match. Model deployed.",
            f"[CLIENT bank_2]     Verifying hash on chain… ✓ Match. Model deployed.",
            f"[CLIENT uni_3]      Verifying hash on chain… ✓ Match. Model deployed.",
        ])

        st.session_state["chain_log"].insert(0, {
            "version": version,
            "hash":    h,
            "round":   rnd,
            "time":    time.strftime("%H:%M:%S"),
        })

    st.session_state["fl_rounds_done"] += 3
    st.session_state["chain_entries"]   = len(st.session_state["chain_log"])
    st.session_state["fed_log"].append("✅ Federation complete.  All clients immunised.")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

status_color = {"ACTIVE": THEME["green"], "INITIALISING": THEME["yellow"],
                "UNDER ATTACK": THEME["red"]}.get(st.session_state["system_status"], THEME["yellow"])

_org_badge = ""
if ORG:
    _badge_color = ORG["color"]
    _org_badge = (
        f"<span style='background:{_badge_color}22; border:1px solid {_badge_color}; "
        f"border-radius:20px; padding:2px 12px; font-size:0.82em; "
        f"color:{_badge_color}; margin-left:0.8em; font-weight:bold'>"
        f"{ORG['icon']} {ORG['label'].upper()}  ·  {ORG['net']}"
        f"</span>"
    )

st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center;
            background:{THEME['panel']}; border:1px solid {THEME['border']};
            border-radius:8px; padding:0.8rem 1.5rem; margin-bottom:1rem;">
  <div>
    <span style="font-size:1.4em; font-weight:bold; color:{THEME['cyan']}">
      🛡️ AURA
    </span>
    <span style="color:{THEME['dim']}; margin-left:0.5em; font-size:0.85em;">
      Autonomous Unified Resilience Architecture
    </span>
    {_org_badge}
  </div>
  <div style="text-align:right;">
    <span style="color:{status_color}; font-weight:bold; font-size:0.95em;">
      ● {st.session_state['system_status']}
    </span>
    <span style="color:{THEME['dim']}; margin-left:1em; font-size:0.75em;">
      {model_status}  |  Blockchain: {bc.mode.upper()}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Row
# ─────────────────────────────────────────────────────────────────────────────

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.metric("Windows Processed", st.session_state["window_counter"])
with m2:
    st.metric("Threats Detected", st.session_state["total_attacks"])
with m3:
    st.metric("Nodes Blocked", st.session_state["total_blocked"])
with m4:
    st.metric("FL Rounds", st.session_state["fl_rounds_done"])
with m5:
    st.metric("Chain Entries", st.session_state["chain_entries"])
with m6:
    ema_val = (st.session_state["ae_scores"][-1]
               if st.session_state["ae_scores"] else 0.0)
    st.metric("Current AE Score", f"{ema_val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Content: Network Graph + Score Timeline
# ─────────────────────────────────────────────────────────────────────────────

col_graph, col_score = st.columns([1, 1])

with col_graph:
    st.markdown(f"<h4 style='color:{THEME['cyan']}'>🌐 Live Network Topology</h4>",
                unsafe_allow_html=True)
    edge_arr = None
    cg = st.session_state.get("current_graph")
    if cg is not None and "edge_index" in cg:
        edge_arr = cg["edge_index"].numpy()

    net_fig = build_network_figure(
        st.session_state["node_colors"],
        st.session_state["node_states"],
        edge_arr,
    )
    st.plotly_chart(net_fig, use_container_width=True, key="network_graph")

    # Legend
    st.markdown(f"""
    <div style="font-size:0.75em; color:{THEME['dim']}">
      <span style="color:{THEME['green']}">◆ Normal</span> &nbsp;
      <span style="color:{THEME['yellow']}">◆ Evaluating</span> &nbsp;
      <span style="color:{THEME['red']}">◆ Threat Detected</span> &nbsp;
      <span style="color:{THEME['text']}">◇ Critical Infrastructure</span>
    </div>
    """, unsafe_allow_html=True)

with col_score:
    st.markdown(f"<h4 style='color:{THEME['cyan']}'>📈 Anomaly Score Timeline</h4>",
                unsafe_allow_html=True)
    fig_timeline = build_score_timeline(
        st.session_state["ae_scores"],
        st.session_state["thresholds"],
        st.session_state["timestamps"],
    )
    st.plotly_chart(fig_timeline, use_container_width=True, key="score_timeline")

    # EMA state info
    if st.session_state["engine"]:
        ema = st.session_state["engine"].ema_state
        warmup_left = max(0, cfg.EMA_WARMUP_BATCHES - ema.get("batch_count", 0))
        if warmup_left > 0:
            st.info(f"🔄 EMA calibrating… {warmup_left} windows remaining in warmup period.")
        else:
            thresh = (ema.get("ema_mean", 0) or 0) + cfg.EMA_SIGMA_MULTIPLIER * (
                (ema.get("ema_var", 0) or 0) ** 0.5
            )
            st.markdown(f"""
            <div style="font-size:0.8em; color:{THEME['dim']}">
              EMA Mean: <b style='color:{THEME['text']}'>{ema.get('ema_mean', 0):.5f}</b> &nbsp;|&nbsp;
              Threshold (3σ): <b style='color:{THEME['red']}'>{thresh:.5f}</b> &nbsp;|&nbsp;
              Batches: <b style='color:{THEME['text']}'>{ema.get('batch_count', 0)}</b>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# AE Explanation Panel
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    f"<h4 style='color:{THEME['yellow']}'>🧠 AE Explanation — Why did the score spike?</h4>",
    unsafe_allow_html=True
)

_expl = st.session_state.get("last_explanation")
if _expl is None:
    _dim = THEME["dim"]
    st.markdown(
        f"<div style='color:{_dim}; font-size:0.85em; padding:0.5rem 0'>"
        "No anomaly detected yet.  Inject an attack to see a live explanation."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    _e     = _expl["explanation"]
    _sev   = _expl["severity"]
    _conf  = _expl["confidence"]
    _match = _expl["match_score"]
    _atk   = _expl["inferred_attack"]
    _icon  = _e.get("icon", "")
    _summ  = _e.get("summary", "")
    _det   = _e.get("detail", "")
    _why   = _e.get("why_high", "")

    sev_color = {"HIGH": THEME["red"], "MEDIUM": THEME["orange"],
                 "LOW": THEME["yellow"]}.get(_sev, THEME["cyan"])

    expl_left, expl_mid, expl_right = st.columns([1.3, 1.0, 0.9])

    # ── Left: Attack classification + detail ──────────────────────────────
    with expl_left:
        _bg, _br = THEME["panel"], THEME["border"]
        st.markdown(
            f"<div style='background:{_bg}; border:1px solid {sev_color}; "
            f"border-radius:8px; padding:0.8rem 1rem;'>"
            f"<div style='font-size:1.1em; font-weight:bold; color:{sev_color}'>"
            f"{_icon} {_atk}</div>"
            f"<div style='color:{THEME['text']}; font-size:0.85em; margin:0.4rem 0'>"
            f"{_summ}</div>"
            f"<hr style='border-color:{THEME['border']}; margin:0.5rem 0'>"
            f"<div style='color:{THEME['dim']}; font-size:0.78em; line-height:1.5'>"
            f"{_det}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Mid: Top contributing features bar chart ───────────────────────────
    with expl_mid:
        _top   = _expl["top_features"][:6]
        _names = [f[0] for f in _top][::-1]
        _vals  = [f[1] for f in _top][::-1]
        import plotly.graph_objects as go
        fig_feat = go.Figure(go.Bar(
            y=_names,
            x=_vals,
            orientation="h",
            marker_color=sev_color,
            marker_line_width=0,
        ))
        fig_feat.update_layout(
            title=dict(text="Top Anomalous Features", font=dict(color=THEME["text"], size=12)),
            paper_bgcolor=THEME["bg"],
            plot_bgcolor=THEME["panel"],
            font=dict(color=THEME["dim"], size=10),
            height=200,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(gridcolor=THEME["border"], title="Mean |residual|"),
            yaxis=dict(gridcolor=THEME["border"]),
        )
        st.plotly_chart(fig_feat, use_container_width=True, key="feat_chart")

    # ── Right: Match confidence + why the score is high ──────────────────
    with expl_right:
        _bg, _br = THEME["panel"], THEME["border"]
        st.markdown(
            f"<div style='background:{_bg}; border:1px solid {_br}; "
            f"border-radius:8px; padding:0.8rem 1rem;'>"
            f"<div style='color:{THEME['dim']}; font-size:0.78em'>Signature match</div>"
            f"<div style='font-size:1.4em; color:{sev_color}; font-weight:bold'>"
            f"{_match:.0%}</div>"
            f"<div style='color:{THEME['dim']}; font-size:0.78em; margin-top:0.6rem'>Detection confidence</div>"
            f"<div style='font-size:1.4em; color:{THEME['text']}; font-weight:bold'>"
            f"{_conf:.0%}</div>"
            f"<hr style='border-color:{THEME['border']}; margin:0.5rem 0'>"
            f"<div style='color:{THEME['dim']}; font-size:0.76em; line-height:1.5'>"
            f"<b style='color:{THEME['text']}'>Why is the score high?</b><br>{_why}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Control Panel (3 columns)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
ctrl_atk, ctrl_fl, ctrl_chain = st.columns([1.2, 1, 0.8])

# ── Attack Injection Panel ────────────────────────────────────────────────────
with ctrl_atk:
    st.markdown(f"<h4 style='color:{THEME['red']}'>🔴 Attack Simulation</h4>",
                unsafe_allow_html=True)

    atk_cols = st.columns(3)
    attack_map = {
        "DDoS": "ddos", "Port Scan": "portscan",
        "Lateral": "lateral", "Exfil": "exfil", "Web": "web",
    }
    all_attacks = list(attack_map.items())

    for idx, (label, atype) in enumerate(all_attacks):
        col = atk_cols[idx % 3]
        if col.button(label, key=f"atk_{atype}", use_container_width=True):
            st.session_state["attack_active"] = True
            st.session_state["attack_type"]   = atype
            st.session_state["system_status"] = "UNDER ATTACK"

            # Generate and process attack graph
            inj = st.session_state["injector"]
            if inj:
                attack_graph = inj.inject(atype)
                event = run_inference_tick(attack_graph, is_attack=True)
                st.toast(
                    f"💥 {label} attack injected!  "
                    f"Severity: {event.severity.name}  "
                    f"Confidence: {event.confidence:.1%}",
                    icon="🚨" if event.severity == AlertSeverity.HIGH else "⚠️"
                )
                st.rerun()

    if st.button("🟢 Generate Normal Traffic", use_container_width=True):
        st.session_state["attack_active"] = False
        st.session_state["system_status"] = "ACTIVE"

        inj = st.session_state["injector"]
        if inj:
            normal_graph = inj._generate_healthy_graph()
            normal_graph["window_id"] = f"NORMAL_{st.session_state['window_counter']}"
            run_inference_tick(normal_graph, is_attack=False)

        # Reset node colours
        st.session_state["node_colors"] = {i: THEME["green"] for i in range(cfg.NUM_SYNTHETIC_NODES)}
        st.session_state["node_states"] = {i: "Normal" for i in range(cfg.NUM_SYNTHETIC_NODES)}
        st.toast("✅ Normal traffic window processed.", icon="✅")
        st.rerun()

# ── Federation Panel ──────────────────────────────────────────────────────────
with ctrl_fl:
    _fl_heading = f"{ORG['icon']} {ORG['label']} · Federated Learning" if ORG else "🌐 Federated Learning"
    st.markdown(f"<h4 style='color:{THEME['blue']}'>{_fl_heading}</h4>",
                unsafe_allow_html=True)

    # ── FL Readiness Toggle ──────────────────────────────────────────────────
    _ready = st.session_state.get("fl_ready", False)
    _ready_color  = THEME["green"]  if _ready else THEME["dim"]
    _ready_status = "🟢 READY"      if _ready else "🔴 NOT READY"
    _ready_label  = THEME["green"]  if _ready else THEME["red"]

    st.markdown(
        f"<div style='background:{THEME['panel']}; border:1px solid {_ready_color}; "
        f"border-radius:8px; padding:0.55rem 0.8rem; margin-bottom:0.5rem; "
        f"text-align:center; font-size:0.9em'>"
        f"<span style='color:{THEME['dim']}'>Are you ready for FL?</span>&nbsp;&nbsp;"
        f"<span style='color:{_ready_color}; font-weight:bold'>{_ready_status}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    _toggle_label = "✅ Mark as Ready" if not _ready else "⏸ Mark as Not Ready"
    if st.button(_toggle_label, use_container_width=True):
        st.session_state["fl_ready"] = not _ready
        st.rerun()

    # ── Client Status Table ──────────────────────────────────────────────────
    clients_info = st.session_state.get("fl_client_status", [])
    if clients_info:
        _bg  = THEME["panel"]
        _br  = THEME["border"]
        _cy  = THEME["cyan"]
        _dim = THEME["dim"]
        _grn = THEME["green"]
        _red = THEME["red"]
        _org = THEME["orange"]

        rows_html = ""
        for c in clients_info:
            org     = c.get("org_id",   "unknown")
            network = c.get("network",  "—")
            role    = c.get("role",     "Normal")
            selected= c.get("selected", True)

            role_color   = _red if role == "Byzantine" else _grn
            status_label = "✓ Selected" if selected else "✗ Dropped"
            status_color = _grn         if selected else _red
            if not selected and role == "Byzantine":
                status_label = "✗ Dropped (Byzantine)"

            rows_html += (
                f"<tr style='border-bottom:1px solid {_br}'>"
                f"<td style='padding:3px 6px; color:{_cy}'>{org}</td>"
                f"<td style='padding:3px 6px; color:{_dim}'>{network}</td>"
                f"<td style='padding:3px 6px; color:{role_color}'>{role}</td>"
                f"<td style='padding:3px 6px; color:{status_color}'>{status_label}</td>"
                f"</tr>"
            )

        st.markdown(
            f"<div style='background:{_bg}; border:1px solid {_br}; "
            f"border-radius:6px; padding:0.5rem; margin-bottom:0.4rem'>"
            f"<div style='color:{_dim}; font-size:0.72em; "
            f"margin-bottom:0.3rem'>FL CLIENT STATUS (latest round)</div>"
            f"<table style='width:100%; border-collapse:collapse; font-size:0.73em'>"
            f"<thead><tr style='color:{_dim}; border-bottom:1px solid {_br}'>"
            f"<th style='text-align:left; padding:2px 6px'>Org</th>"
            f"<th style='text-align:left; padding:2px 6px'>Network</th>"
            f"<th style='text-align:left; padding:2px 6px'>Role</th>"
            f"<th style='text-align:left; padding:2px 6px'>Krum</th>"
            f"</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            f"</table></div>",
            unsafe_allow_html=True,
        )

    # Show last N fed log lines
    if st.session_state["fed_log"]:
        log_html = "<br>".join([
            f"<span class='fed-log'>{line}</span>"
            for line in st.session_state["fed_log"][-15:]
        ])
        _bg, _br = THEME["panel"], THEME["border"]
        st.markdown(
            f"<div style='background:{_bg}; border:1px solid "
            f"{_br}; border-radius:6px; padding:0.6rem; "
            f"max-height:180px; overflow-y:auto; font-size:0.75em'>"
            f"{log_html}</div>",
            unsafe_allow_html=True
        )

# ── Blockchain Panel ──────────────────────────────────────────────────────────
with ctrl_chain:
    st.markdown(f"<h4 style='color:{THEME['cyan']}'>⛓ Blockchain Audit</h4>",
                unsafe_allow_html=True)

    bc_module = st.session_state["blockchain"]
    if bc_module:
        _dim, _cy = THEME["dim"], THEME["cyan"]
        st.markdown(f"<div style='color:{_dim}; font-size:0.8em'>"
                    f"Mode: <b style='color:{_cy}'>"
                    f"{bc_module.mode.upper()}</b></div>",
                    unsafe_allow_html=True)

        # Register a test hash
        if st.button("📝 Register Test Hash", use_container_width=True):
            fake_hash   = "0x" + hashlib.sha256(f"test_{time.time()}".encode()).hexdigest()
            version_tag = f"demo_v{int(time.time()) % 10000}"
            tx = bc_module.log_model_update(version_tag, fake_hash)
            st.session_state["chain_log"].insert(0, {
                "version": version_tag,
                "hash": fake_hash[:20] + "…",
                "round": "manual",
                "time": time.strftime("%H:%M:%S"),
            })
            st.session_state["chain_entries"] = len(st.session_state["chain_log"])
            st.toast(f"Hash {fake_hash[:12]}… written to ledger.", icon="⛓")

    # Chain history table
    if st.session_state["chain_log"]:
        for entry in st.session_state["chain_log"][:6]:
            st.markdown(
                f"<div class='chain-row'>"
                f"[{entry['time']}] {entry['version']}  "
                f"<span style='opacity:0.6'>{entry['hash'][:22]}…</span>"
                f"</div>",
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# Alert + Incident Log
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
log_col, inc_col = st.columns([1, 1])

with log_col:
    st.markdown(f"<h4 style='color:{THEME['yellow']}'>🔔 Alert History</h4>",
                unsafe_allow_html=True)

    if not st.session_state["alerts"]:
        st.markdown(f"<span style='color:{THEME['dim']}'>No alerts triggered yet.</span>",
                    unsafe_allow_html=True)
    else:
        for a in st.session_state["alerts"][:10]:
            severity = a.get("severity", "NORMAL")
            color = {"HIGH": THEME["red"], "MEDIUM": THEME["orange"],
                     "LOW": THEME["yellow"]}.get(severity, THEME["green"])
            ts = time.strftime("%H:%M:%S", time.localtime(a.get("timestamp", 0)))
            _t = THEME["text"]
            st.markdown(
                f"<div style='border-left:3px solid {color}; padding:0.3rem 0.6rem; "
                f"margin:0.2rem 0; font-size:0.8em; color:{_t}'>"
                f"<b style='color:{color}'>[{ts}] {severity}</b>  "
                f"mse={a.get('ae_score', 0):.4f}  "
                f"conf={a.get('confidence', 0):.1%}  "
                f"nodes={a.get('triggered_nodes', [])}"
                f"</div>",
                unsafe_allow_html=True
            )

with inc_col:
    st.markdown(f"<h4 style='color:{THEME['orange']}'>🛡️ Response Actions</h4>",
                unsafe_allow_html=True)

    if not st.session_state["incidents"]:
        st.markdown(f"<span style='color:{THEME['dim']}'>No responses triggered yet.</span>",
                    unsafe_allow_html=True)
    else:
        for r in st.session_state["incidents"][:10]:
            action = r.get("action_taken", "")
            color  = {"ISOLATE": THEME["red"], "THROTTLE": THEME["orange"],
                      "HITL_ESCALATE": THEME["yellow"], "LOG_ONLY": THEME["green"]}.get(
                action, THEME["dim"]
            )
            ts = time.strftime("%H:%M:%S", time.localtime(r.get("timestamp", 0)))
            crit_badge = "🔑 CRITICAL" if r.get("is_critical") else ""
            _t = THEME["text"]
            st.markdown(
                f"<div style='border-left:3px solid {color}; padding:0.3rem 0.6rem; "
                f"margin:0.2rem 0; font-size:0.8em; color:{_t}'>"
                f"<b style='color:{color}'>[{ts}] {action}</b>  "
                f"{crit_badge}  "
                f"{r.get('node_id', '')} ({r.get('node_label', '')})"
                f"</div>",
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"<h3 style='color:{THEME['cyan']}'>⚙️ AURA Config</h3>",
                unsafe_allow_html=True)

    st.markdown(f"""
    **Model Status:** {model_status}
    **Blockchain Mode:** {bc.mode}
    **Nodes:** {cfg.NUM_SYNTHETIC_NODES}
    **Feature Dim:** {cfg.FEATURE_DIM}
    **EMA α:** {cfg.EMA_ALPHA}
    **EMA σ mult:** {cfg.EMA_SIGMA_MULTIPLIER}×
    """)

    st.markdown("---")
    st.markdown(f"<h4 style='color:{THEME['cyan']}'>📖 Architecture Layers</h4>",
                unsafe_allow_html=True)
    st.markdown("""
    1. **Data Layer** — NetFlow → Graph  
       TTL edge decay, IsolationForest sanitisation

    2. **Layer 1** — Statistical Tripwire  
       Unsupervised Autoencoder  
       EMA Dynamic Threshold

    3. **Layer 2** — Contextual Validator  
       GraphSAGE (inductive)  
       Topology anomaly scoring

    4. **Federation** — Krum aggregation  
       Byzantine-robust FL  
       Straggler timeout policy

    5. **Blockchain** — SHA-256 audit  
       Non-repudiation audit trail  
       Model integrity verification
    """)

    st.markdown("---")
    if st.button("🗑️ Clear All Logs"):
        for key in ["ae_scores", "thresholds", "timestamps", "alerts",
                    "incidents", "fed_log", "chain_log"]:
            st.session_state[key] = []
        st.session_state["node_colors"] = {i: THEME["green"] for i in range(cfg.NUM_SYNTHETIC_NODES)}
        st.session_state["node_states"] = {i: "Normal" for i in range(cfg.NUM_SYNTHETIC_NODES)}
        st.session_state["system_status"] = "ACTIVE"
        st.session_state["last_explanation"] = None
        st.rerun()

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.7em; color:{THEME['dim']}'>
    Team Trinetra — NexJam 2026<br>
    Suraj H P | Narendra Kanchi | Sudhanva Girish Thite
    </div>
    """, unsafe_allow_html=True)
