"""
fl_server_dashboard.py — AURA Federated Learning Server Dashboard
==================================================================

Dedicated server-side console showing:
  • All three org clients (Hospital / Bank / University)
  • Step-by-step FL pipeline animation (collect → Krum → aggregate → mint → broadcast)
  • Blockchain hash minting live feed
  • Per-round history table with Krum scores
  • Client hash verification outcome
"""

import hashlib
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────
THEME = {
    "bg":       "#0d1117",
    "panel":    "#161b22",
    "panel2":   "#1c2430",
    "border":   "#30363d",
    "cyan":     "#58d1e8",
    "green":    "#3fb950",
    "red":      "#f85149",
    "orange":   "#d29922",
    "blue":     "#388bfd",
    "purple":   "#bc8cff",
    "yellow":   "#e3b341",
    "dim":      "#8b949e",
    "text":     "#c9d1d9",
}

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "AURA · FL Server Console",
    page_icon   = "⚙️",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  html, body, [class*="st-"] {{
    background-color: {THEME['bg']};
    color: {THEME['text']};
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }}
  .block-container {{ padding: 1rem 1.5rem; }}
  h1, h2, h3, h4 {{ color: {THEME['cyan']}; }}

  /* Client card */
  .client-card {{
    background: {THEME['panel']};
    border: 1px solid {THEME['border']};
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.4rem;
  }}
  .client-card.selected  {{ border-color: {THEME['green']}; }}
  .client-card.dropped   {{ border-color: {THEME['red']}; }}
  .client-card.byzantine {{ border-color: {THEME['orange']}; }}
  .client-card.idle      {{ border-color: {THEME['border']}; }}

  /* Pipeline step */
  .pipe-step {{
    background: {THEME['panel2']};
    border: 1px solid {THEME['border']};
    border-radius: 8px;
    padding: 0.5rem 0.7rem;
    text-align: center;
    font-size: 0.78em;
    transition: border-color 0.2s;
  }}
  .pipe-step.active  {{ border-color: {THEME['cyan']}; background: #1a2a38; }}
  .pipe-step.done    {{ border-color: {THEME['green']}; background: #0f2117; }}
  .pipe-step.pending {{ opacity: 0.45; }}

  /* Hash card */
  .hash-row {{
    background: {THEME['panel']};
    border-left: 3px solid {THEME['purple']};
    border-radius: 4px;
    padding: 0.35rem 0.6rem;
    margin: 0.25rem 0;
    font-size: 0.74em;
    font-family: monospace;
  }}
  .hash-row.final {{ border-left-color: {THEME['green']}; }}

  /* Round history row */
  .round-row {{
    display: flex; gap: 0.5rem;
    padding: 0.25rem 0.5rem;
    border-bottom: 1px solid {THEME['border']};
    font-size: 0.76em;
  }}
  .verify-ok   {{ color: {THEME['green']}; }}
  .verify-warn {{ color: {THEME['red']}; }}
  div[data-testid="stMetricValue"] {{ font-size: 1.6em; color: {THEME['cyan']}; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────

_ORGS = [
    {"id": "org_hospital_1",    "label": "Hospital",    "net": "192.168.1.0/24",  "role": "Normal"},
    {"id": "org_bank_2",        "label": "Bank",        "net": "10.0.1.0/24",     "role": "Byzantine"},
    {"id": "org_university_3",  "label": "University",  "net": "172.16.1.0/24",   "role": "Normal"},
]

_PIPE_STEPS = [
    ("📥", "Collect\nWeights"),
    ("🔬", "Krum\nFilter"),
    ("🧮", "Aggregate"),
    ("⛓",  "Mint\nHash"),
    ("📡", "Broadcast\n+ Verify"),
]

def _init():
    defaults = {
        "fl_running":       False,
        "fl_done":          False,
        "round_results":    [],       # list of per-round dicts
        "hash_ledger":      [],       # blockchain entries (final rounds)
        "hash_local":       [],       # intermediate hash records
        "client_cards":     {o["id"]: {"status": "idle", "round": 0,
                                        "selected": None, "verified": None}
                             for o in _ORGS},
        "pipe_state":       [0] * len(_PIPE_STEPS),  # 0=pending,1=active,2=done
        "current_round":    0,
        "total_rounds":     cfg.FL_NUM_ROUNDS,
        "fl_log":           [],
        "krum_scores_hist": [],  # list of dicts per round
        "global_hash":      None,
        "global_version":   None,
        "verify_results":   {},   # org_id → True/False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ─────────────────────────────────────────────────────────────────────────────
# Helper — log line
# ─────────────────────────────────────────────────────────────────────────────
def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state["fl_log"].insert(0, f"[{ts}] {msg}")
    st.session_state["fl_log"] = st.session_state["fl_log"][:120]


# ─────────────────────────────────────────────────────────────────────────────
# FL Simulation with step-by-step UI updates
# ─────────────────────────────────────────────────────────────────────────────

def run_fl_with_animation(pipe_ph, card_placeholders, log_ph, ledger_ph,
                           metrics_ph, round_hist_ph):
    """
    Drive the FL simulation round-by-round, updating Streamlit placeholders
    at each pipeline step so the operator sees live progress.
    """
    from aura.fl_server import hash_model_weights, krum_select, krum_aggregate, KrumFedAURA
    from aura.fl_client import create_mock_clients
    from aura.models import AURAModelBundle as MB
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitIns
    import numpy as np

    st.session_state["fl_running"]    = True
    st.session_state["fl_done"]       = False
    st.session_state["round_results"] = []
    st.session_state["hash_ledger"]   = []
    st.session_state["hash_local"]    = []
    st.session_state["fl_log"]        = []
    st.session_state["krum_scores_hist"] = []
    st.session_state["verify_results"] = {}
    st.session_state["current_round"] = 0

    # Reset client cards
    for o in _ORGS:
        st.session_state["client_cards"][o["id"]] = {
            "status": "idle", "round": 0, "selected": None, "verified": None,
        }

    n_rounds = cfg.FL_NUM_ROUNDS
    st.session_state["total_rounds"] = n_rounds

    # Build real clients + strategy (mirrors run_federation_simulation)
    clients  = create_mock_clients(n_clients=3, n_samples=300)
    strategy = KrumFedAURA(blockchain_module=None, num_rounds=n_rounds)

    global_model  = MB()
    global_params = [p.detach().cpu().numpy() for p in global_model.parameters()]

    _log(f"Federation started — {n_rounds} rounds, {len(clients)} clients")

    for rnd in range(1, n_rounds + 1):
        st.session_state["current_round"] = rnd
        is_final = (rnd == n_rounds)
        _log(f"━━━  Round {rnd}/{n_rounds}  ━━━")

        # ── STEP 0: Collect Weights ───────────────────────────────────────
        _set_pipe(0, "active"); _render_pipe(pipe_ph); _render_clients(card_placeholders)

        fit_results = []
        for i, client in enumerate(clients):
            org_id = _ORGS[i]["id"]
            st.session_state["client_cards"][org_id]["status"] = "sending"
            st.session_state["client_cards"][org_id]["round"]  = rnd
            _render_clients(card_placeholders)

            fit_ins = FitIns(
                parameters = ndarrays_to_parameters(global_params),
                config     = {"local_epochs": 3, "round": rnd},
            )
            fit_res = client.fit(fit_ins)
            fit_results.append((None, fit_res))
            raw_loss = fit_res.metrics.get('train_loss', None)
            loss_str = f"{raw_loss:.4f}" if isinstance(raw_loss, (int, float)) else str(raw_loss)
            _log(f"  [{_ORGS[i]['label']}] Weights received — loss={loss_str}")

        time.sleep(0.35)
        _set_pipe(0, "done")

        # ── STEP 1: Krum Filter ───────────────────────────────────────────
        _set_pipe(1, "active"); _render_pipe(pipe_ph)
        _log(f"  [SERVER] Running Krum filter on {len(clients)} updates …")

        client_updates = [parameters_to_ndarrays(r.parameters) for _, r in fit_results]

        # Compute flat vectors + scores manually for display
        flat = [np.concatenate([p.flatten() for p in u]) for u in client_updates]
        n, k = len(flat), max(1, len(flat) - cfg.KRUM_NUM_TO_SELECT - 2)
        scores = []
        for i in range(n):
            dists = sorted([float(np.sum((flat[i] - flat[j])**2))
                            for j in range(n) if j != i])
            scores.append(sum(dists[:k]))

        selected_indices = krum_select(client_updates, cfg.KRUM_NUM_TO_SELECT)
        dropped_indices  = [i for i in range(n) if i not in selected_indices]

        st.session_state["krum_scores_hist"].append({
            "round":    rnd,
            "scores":   [round(s, 2) for s in scores],
            "selected": selected_indices,
            "dropped":  dropped_indices,
        })

        for i, org in enumerate(_ORGS):
            is_sel = i in selected_indices
            st.session_state["client_cards"][org["id"]]["status"]   = "selected" if is_sel else "dropped"
            st.session_state["client_cards"][org["id"]]["selected"] = is_sel
        _render_clients(card_placeholders)

        sel_labels   = [_ORGS[i]["label"] for i in selected_indices]
        drop_labels  = [_ORGS[i]["label"] for i in dropped_indices]
        _log(f"  [KRUM] Selected: {sel_labels}  |  Dropped: {drop_labels}")
        _log(f"  [KRUM] Scores → {['%.1f'%s for s in scores]}")
        time.sleep(0.35)
        _set_pipe(1, "done")

        # ── STEP 2: Aggregate ─────────────────────────────────────────────
        _set_pipe(2, "active"); _render_pipe(pipe_ph)
        selected_updates = [client_updates[i] for i in selected_indices]
        aggregated       = krum_aggregate(selected_updates)
        global_params    = aggregated

        model_version = f"final_v{rnd}" if is_final else f"v{rnd}.{rnd}"
        _log(f"  [SERVER] Global model {model_version} aggregated from "
             f"{len(selected_indices)} updates")
        time.sleep(0.3)
        _set_pipe(2, "done")

        # ── STEP 3: Mint Hash ─────────────────────────────────────────────
        _set_pipe(3, "active"); _render_pipe(pipe_ph)

        model_hash = hash_model_weights(aggregated)
        st.session_state["global_hash"]    = model_hash
        st.session_state["global_version"] = model_version

        if is_final:
            # Write to registry (mirrors fl_server logic)
            from pathlib import Path
            import json
            reg_path = Path(cfg.LOGS_DIR) / "hash_registry.json"
            reg_path.parent.mkdir(parents=True, exist_ok=True)
            reg_path.write_text(json.dumps({model_version: model_hash}, indent=2))

            st.session_state["hash_ledger"].append({
                "round": rnd, "version": model_version,
                "hash":  model_hash, "time": time.strftime("%H:%M:%S"),
                "final": True,
            })
            _log(f"  [BLOCKCHAIN] ✅ Hash MINTED — {model_version}")
            _log(f"  [BLOCKCHAIN]   SHA-256: {model_hash[:32]}…")
        else:
            st.session_state["hash_local"].append({
                "round": rnd, "version": model_version,
                "hash":  model_hash, "time": time.strftime("%H:%M:%S"),
            })
            _log(f"  [HASH] Intermediate hash recorded (not minted) — {model_hash[:20]}…")

        _render_ledger(ledger_ph)
        time.sleep(0.35)
        _set_pipe(3, "done")

        # ── STEP 4: Broadcast + Verify ────────────────────────────────────
        _set_pipe(4, "active"); _render_pipe(pipe_ph)
        _log(f"  [SERVER] Broadcasting global model {model_version} to all clients …")
        time.sleep(0.2)

        if is_final:
            client_received_hash = hash_model_weights(global_params)
            for i, org in enumerate(_ORGS):
                match = (client_received_hash == model_hash)
                st.session_state["verify_results"][org["id"]] = match
                st.session_state["client_cards"][org["id"]]["verified"] = match
                status = "✓ MATCH — deployed" if match else "✗ MISMATCH — rejected"
                _log(f"  [{_ORGS[i]['label']}] Verify: {status}")
            _render_clients(card_placeholders)
        else:
            _log(f"  [CLIENTS] Round {rnd} model received. "
                 f"Verification on final round only.")

        _set_pipe(4, "done"); _render_pipe(pipe_ph)

        # ── Persist round result ──────────────────────────────────────────
        st.session_state["round_results"].append({
            "round":          rnd,
            "model_version":  model_version,
            "model_hash":     model_hash,
            "krum_selected":  len(selected_indices),
            "krum_dropped":   len(dropped_indices),
            "krum_scores":    [round(s, 2) for s in scores],
            "is_final":       is_final,
        })

        _render_metrics(metrics_ph)
        _render_round_hist(round_hist_ph)
        time.sleep(0.4)

        # Reset pipe for next round
        if rnd < n_rounds:
            st.session_state["pipe_state"] = [0] * len(_PIPE_STEPS)

    st.session_state["fl_running"] = False
    st.session_state["fl_done"]    = True
    _log(f"✅ Federation complete — {n_rounds} rounds finished")
    _log(f"   Final hash: {st.session_state['global_hash'][:24] if st.session_state['global_hash'] else 'N/A'}…")


# ─────────────────────────────────────────────────────────────────────────────
# Renderers — write into placeholders
# ─────────────────────────────────────────────────────────────────────────────

def _set_pipe(idx: int, state: str):
    mapping = {"pending": 0, "active": 1, "done": 2}
    st.session_state["pipe_state"][idx] = mapping[state]


def _render_pipe(ph):
    states = st.session_state["pipe_state"]
    state_cls = ["pending", "active", "done"]
    state_ico = ["⏳", "🔄", "✅"]

    cols_html = ""
    for i, (icon, label) in enumerate(_PIPE_STEPS):
        cls = state_cls[states[i]]
        ico = state_ico[states[i]]
        arrow = " <span style='color:#8b949e; font-size:1.2em'>→</span> " if i < len(_PIPE_STEPS) - 1 else ""
        cols_html += (
            f"<span class='pipe-step {cls}' style='display:inline-block; "
            f"min-width:90px; margin:0 2px'>"
            f"{ico} {icon}<br><b>{label}</b></span>{arrow}"
        )

    bg, br = THEME["panel"], THEME["border"]
    ph.markdown(
        f"<div style='background:{bg}; border:1px solid {br}; border-radius:8px; "
        f"padding:0.65rem 1rem; text-align:center; font-size:0.8em'>"
        f"{cols_html}</div>",
        unsafe_allow_html=True,
    )


def _render_clients(card_phs):
    cards = st.session_state["client_cards"]
    status_label = {
        "idle":     ("Idle",          THEME["dim"],   "idle"),
        "sending":  ("Sending…",      THEME["yellow"], "idle"),
        "selected": ("✓ Selected",    THEME["green"],  "selected"),
        "dropped":  ("✗ Dropped",     THEME["red"],    "dropped"),
    }
    for i, org in enumerate(_ORGS):
        c    = cards[org["id"]]
        raw  = c.get("status", "idle")
        lbl, color, css = status_label.get(raw, ("Idle", THEME["dim"], "idle"))
        role_color = THEME["orange"] if org["role"] == "Byzantine" else THEME["green"]
        vfy = c.get("verified")
        vfy_html = ""
        _vfy_grn = THEME["green"]
        _vfy_red = THEME["red"]
        if vfy is True:
            vfy_html = f"<div style='color:{_vfy_grn}; font-size:0.8em'>&#9939; Hash verified &#10003;</div>"
        elif vfy is False:
            vfy_html = f"<div style='color:{_vfy_red}; font-size:0.8em'>&#9939; Hash MISMATCH &#10007;</div>"

        ip_map = {"Normal": "✓ Normal", "Byzantine": "⚠ Byzantine"}

        card_phs[i].markdown(
            f"<div class='client-card {css}'>"
            f"<div style='font-size:1.0em; font-weight:bold; color:{THEME['cyan']}'>"
            f"{['🏥','🏦','🎓'][i]} {org['label']}</div>"
            f"<div style='font-size:0.77em; color:{THEME['dim']}'>{org['id']}</div>"
            f"<div style='font-size:0.77em; color:{THEME['dim']}'>🌐 {org['net']}</div>"
            f"<div style='font-size:0.78em; color:{role_color}; margin-top:2px'>"
            f"{ip_map[org['role']]}</div>"
            f"<div style='font-size:0.85em; color:{color}; margin-top:4px'>"
            f"<b>{lbl}</b></div>"
            f"{vfy_html}"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_ledger(ph):
    entries = st.session_state["hash_ledger"]
    local   = st.session_state["hash_local"]
    bg, br = THEME["panel"], THEME["border"]

    rows = ""
    for e in reversed(entries):
        rows += (
            f"<div class='hash-row final'>"
            f"<span style='color:{THEME['green']}'>⛓ MINTED</span> "
            f"<b style='color:{THEME['cyan']}'>{e['version']}</b>  "
            f"<span style='color:{THEME['text']}'>{e['hash'][:28]}…</span>  "
            f"<span style='color:{THEME['dim']}'>[{e['time']}]</span>"
            f"</div>"
        )
    for e in reversed(local[-4:]):
        rows += (
            f"<div class='hash-row'>"
            f"<span style='color:{THEME['dim']}'>📋 local </span> "
            f"<b style='color:{THEME['dim']}'>{e['version']}</b>  "
            f"<span style='color:{THEME['dim']}'>{e['hash'][:28]}…</span>  "
            f"<span style='color:{THEME['dim']}'>[{e['time']}]</span>"
            f"</div>"
        )

    if not rows:
        rows = f"<div style='color:{THEME['dim']}; font-size:0.8em'>No hashes yet …</div>"

    ph.markdown(
        f"<div style='background:{bg}; border:1px solid {br}; border-radius:8px; "
        f"padding:0.65rem; min-height:80px'>"
        f"<div style='color:{THEME['dim']}; font-size:0.72em; margin-bottom:0.4rem'>"
        f"BLOCKCHAIN LEDGER</div>{rows}</div>",
        unsafe_allow_html=True,
    )


def _render_metrics(ph):
    rr = st.session_state["round_results"]
    if not rr:
        return
    last = rr[-1]
    c1, c2, c3, c4 = ph.columns(4)
    c1.metric("Rounds Done",  f"{len(rr)} / {st.session_state['total_rounds']}")
    c2.metric("Krum Selected", f"{last.get('krum_selected','?')} / 3")
    c3.metric("Krum Dropped",  str(last.get("krum_dropped", 0)))
    status = "✅ Done" if st.session_state["fl_done"] else "🔄 Running"
    c4.metric("Status", status)


def _render_round_hist(ph):
    rr = st.session_state["round_results"]
    bg, br = THEME["panel"], THEME["border"]
    if not rr:
        return

    header = (
        f"<tr style='color:{THEME['dim']}; border-bottom:1px solid {br}'>"
        f"<th style='padding:3px 8px'>Round</th>"
        f"<th>Version</th>"
        f"<th>✓ Kept</th>"
        f"<th>✗ Dropped</th>"
        f"<th>Krum Scores</th>"
        f"<th>Hash (truncated)</th>"
        f"<th>On-Chain</th>"
        f"</tr>"
    )
    rows = ""
    for r in rr:
        chain_mark = (f"<span style='color:{THEME['green']}'>⛓ MINTED</span>"
                      if r["is_final"]
                      else f"<span style='color:{THEME['dim']}'>— local</span>")
        scores_str = "  ".join([f"{s:.1f}" for s in r.get("krum_scores", [])])
        rows += (
            f"<tr style='border-bottom:1px solid {br}; font-size:0.77em'>"
            f"<td style='padding:3px 8px; color:{THEME['cyan']}'>{r['round']}</td>"
            f"<td style='color:{THEME['text']}'>{r['model_version']}</td>"
            f"<td style='color:{THEME['green']}'>{r['krum_selected']}</td>"
            f"<td style='color:{THEME['red']}'>{r['krum_dropped']}</td>"
            f"<td style='color:{THEME['dim']}; font-family:monospace'>{scores_str}</td>"
            f"<td style='color:{THEME['dim']}; font-family:monospace'>{r['model_hash'][:22]}…</td>"
            f"<td>{chain_mark}</td>"
            f"</tr>"
        )

    ph.markdown(
        f"<div style='background:{bg}; border:1px solid {br}; border-radius:8px; "
        f"padding:0.65rem; overflow-x:auto'>"
        f"<div style='color:{THEME['dim']}; font-size:0.72em; margin-bottom:0.4rem'>"
        f"ROUND HISTORY</div>"
        f"<table style='width:100%; border-collapse:collapse'>"
        f"<thead>{header}</thead><tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _render_log(ph):
    lines = st.session_state["fl_log"][:24]
    bg, br = THEME["panel"], THEME["border"]
    body = "<br>".join(
        f"<span style='color:{THEME['dim']}'>{l}</span>" for l in lines
    ) or f"<span style='color:{THEME['dim']}'>Waiting for FL run…</span>"
    ph.markdown(
        f"<div style='background:{bg}; border:1px solid {br}; border-radius:8px; "
        f"padding:0.65rem; max-height:240px; overflow-y:auto; font-size:0.75em; "
        f"font-family:monospace'>{body}</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

# ── Header ───────────────────────────────────────────────────────────────────
rnd_cur   = st.session_state["current_round"]
rnd_total = st.session_state["total_rounds"]
run_state = ("🔄 RUNNING" if st.session_state["fl_running"]
             else ("✅ COMPLETE" if st.session_state["fl_done"] else "⏹ IDLE"))
run_color = (THEME["yellow"]  if st.session_state["fl_running"]
             else (THEME["green"] if st.session_state["fl_done"] else THEME["dim"]))

st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center;
            background:{THEME['panel']}; border:1px solid {THEME['border']};
            border-radius:10px; padding:0.8rem 1.5rem; margin-bottom:0.8rem">
  <div>
    <span style="font-size:1.5em; font-weight:bold; color:{THEME['cyan']}">
      ⚙️ AURA · FL Server Console
    </span>
    <span style="color:{THEME['dim']}; margin-left:0.8em; font-size:0.82em">
      Krum-Aggregated Federated Learning  ·  Blockchain-Audited
    </span>
  </div>
  <div style="text-align:right">
    <span style="color:{run_color}; font-weight:bold; font-size:1.0em">
      ● {run_state}
    </span>
    <span style="color:{THEME['dim']}; margin-left:1em; font-size:0.78em">
      Round {rnd_cur}/{rnd_total}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics row (placeholder — updated during run) ───────────────────────────
metrics_ph = st.empty()

# Pre-fill initial metrics display
with metrics_ph.container():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rounds Done",   f"0 / {rnd_total}")
    c2.metric("Krum Selected", "— / 3")
    c3.metric("Krum Dropped",  "—")
    c4.metric("Status",        run_state)

st.markdown("---")

# ── Client Cards Row ─────────────────────────────────────────────────────────
st.markdown(f"<h4 style='color:{THEME['blue']}'>🖥 Participating Clients</h4>",
            unsafe_allow_html=True)
card_cols = st.columns(3)
card_placeholders = []
for col in card_cols:
    card_placeholders.append(col.empty())

# Initial render
_render_clients(card_placeholders)

st.markdown("---")

# ── Pipeline Animation ────────────────────────────────────────────────────────
st.markdown(f"<h4 style='color:{THEME['blue']}'>⚡ FL Pipeline</h4>",
            unsafe_allow_html=True)
pipe_ph = st.empty()
_render_pipe(pipe_ph)

st.markdown("---")

# ── Control ──────────────────────────────────────────────────────────────────
btn_col, info_col = st.columns([1, 3])
with btn_col:
    run_btn = st.button(
        "▶ Run Federated Learning",
        use_container_width=True,
        disabled=st.session_state["fl_running"],
        type="primary",
    )
with info_col:
    st.markdown(
        f"<div style='color:{THEME['dim']}; font-size:0.82em; padding-top:0.6rem'>"
        f"3 clients  ·  {rnd_total} rounds  ·  Krum(n=3, keep=2)  "
        f"·  1 blockchain mint (final round only)"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Blockchain Ledger ─────────────────────────────────────────────────────────
st.markdown(f"<h4 style='color:{THEME['purple']}'>⛓ Blockchain Ledger</h4>",
            unsafe_allow_html=True)
ledger_ph = st.empty()
_render_ledger(ledger_ph)

st.markdown("---")

# ── Round History ─────────────────────────────────────────────────────────────
st.markdown(f"<h4 style='color:{THEME['cyan']}'>📊 Round History</h4>",
            unsafe_allow_html=True)
round_hist_ph = st.empty()
_render_round_hist(round_hist_ph)

st.markdown("---")

# ── FL Log ────────────────────────────────────────────────────────────────────
st.markdown(f"<h4 style='color:{THEME['dim']}'>📋 Server Log</h4>",
            unsafe_allow_html=True)
log_ph = st.empty()
_render_log(log_ph)

# ─────────────────────────────────────────────────────────────────────────────
# Run button handler
# ─────────────────────────────────────────────────────────────────────────────

if run_btn and not st.session_state["fl_running"]:
    run_fl_with_animation(
        pipe_ph            = pipe_ph,
        card_placeholders  = card_placeholders,
        log_ph             = log_ph,
        ledger_ph          = ledger_ph,
        metrics_ph         = metrics_ph,
        round_hist_ph      = round_hist_ph,
    )
    _render_log(log_ph)
    st.rerun()

# ── Auto-refresh log while idle ───────────────────────────────────────────────
if st.session_state["fl_done"] and st.session_state["round_results"]:
    _render_round_hist(round_hist_ph)
    _render_ledger(ledger_ph)
    _render_metrics(metrics_ph)
