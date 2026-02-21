# AURA — Adaptive Unified Response Architecture

> **Zero-Day Network Threat Detection using Graph Neural Networks, Federated Learning & Blockchain Audit**
> 
> Team Trinetra · NEXJEM Hackathon 2026

---

## What is AURA?

AURA is a production-grade, privacy-preserving network intrusion detection system that combines:

- **GraphSAGE (Inductive GNN)** — models network topology as a graph; detects anomalies in traffic relationships, not just individual packet stats
- **Flow Autoencoder** — reconstruction-error based anomaly scoring; learns what "normal" looks like and flags deviations
- **EMA Dynamic Thresholding** — self-calibrating threshold using Exponential Moving Average (no hardcoded cutoffs)
- **Federated Learning with Krum Aggregation** — multiple network segments train locally; only model weights are shared, never raw traffic (privacy-preserving)
- **Blockchain Audit Log** — every global model update is SHA-256 hashed and written immutably to ledger (tamper-evident supply-chain defence)
- **3-Tier Automated Response** — LOG → THROTTLE → ISOLATE based on severity and node criticality

---

## Architecture

```
Raw Network Traffic (CICIDS2017)
          │
          ▼
┌─────────────────────────────┐
│  Phase 1: Data Ingestion    │  IsolationForest baseline sanitisation
│  TTL Edge Decay             │  Synthetic node topology mapping
│  MinMax Feature Scaling     │  Streaming graph windows
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 2: Anomaly Detection │
│  Layer 1: FlowAutoencoder   │  78→64→32→16→32→64→78 (MSE score)
│  Layer 2: AuraSTGNN         │  GraphSAGE 78→64→32→1 (topology score)
│  EMA Threshold (3σ)         │  Adaptive, warms up over 50 windows
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 3: Federated Learn.  │  Flower (flwr) framework
│  Krum Aggregation           │  Byzantine-fault tolerant (drops poisoned updates)
│  Straggler Timeout (30s)    │  Gradient clipping (norm=1.0)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 4: Response Engine   │
│  LOW    → LOG_ONLY          │
│  MEDIUM → THROTTLE + HITL   │  tc 10kbps + human analyst alert
│  HIGH + Critical → THROTTLE │  Never auto-isolates DC/SCADA/DB
│  HIGH + Standard → ISOLATE  │  iptables DROP (blast-radius contained)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 5: Blockchain Audit  │  SHA-256 model hash per FL round
│  ModelRegistry.sol          │  Solidity 0.8.19 smart contract
│  Local fallback (no Ganache)│  JSONL ledger if blockchain offline
└─────────────────────────────┘
```

---

## Project Structure

```
NEXJEM/
├── aura/
│   ├── __init__.py
│   ├── data_loader.py       # CICIDS2017 pipeline, IsolationForest sanitisation
│   ├── models.py            # FlowAutoencoder + AuraSTGNN (manual SAGEConv)
│   ├── detector.py          # EMA dynamic thresholding, cascade L1→L2
│   ├── response_engine.py   # 3-tier policy engine, HITL, iptables simulation
│   ├── fl_client.py         # Flower federated learning client
│   ├── fl_server.py         # Krum aggregation, straggler policy
│   ├── blockchain.py        # Web3 + local fallback audit logger
│   └── attack_injector.py   # 5 attack profiles for red-team simulation
├── contracts/
│   └── ModelRegistry.sol    # Solidity smart contract for model hash registry
├── dashboard.py             # Streamlit live demo dashboard
├── train.py                 # Two-phase training pipeline
├── run.py                   # Quick-start launcher
├── verify_chain.py          # Blockchain integrity verifier
├── config.py                # All hyperparameters and paths
├── requirements.txt
└── README.md
```

---

## Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017

- **Download:** https://www.unb.ca/cic/datasets/ids-2017.html
- **Variant used:** `MachineLearningCSV` (78 statistical flow features + Label)
- **Place files in:** `CSV's/MachineLearningCVE/`

The dataset is **not included** in this repository due to size (several GB).

Attack types covered: DDoS, Port Scan, Brute Force, Web Attacks (XSS, SQLi), Infiltration, Botnet, DoS

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/SenseiSuraj24/TRINETRA---NEXJEM.git
cd TRINETRA---NEXJEM
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Train the models (quick — ~2 min)

```bash
python run.py train --quick
```

### 4. Launch the dashboard

```bash
python run.py dashboard
# Open http://localhost:8501
```

### 5. Run sanity tests

```bash
python run.py test
```

### 6. CLI pipeline demo (no browser)

```bash
python run.py demo
```

### 7. Verify blockchain integrity

```bash
python verify_chain.py
```

---

## Key Technical Decisions

| Decision | Why |
|---|---|
| **GraphSAGE over GCN** | Inductive — detects threats on new/unseen nodes without retraining |
| **Manual SAGEConv** | No torch_geometric dependency; implemented via `torch.scatter_add_` — more portable |
| **EMA threshold over static** | Network baseline drifts (Monday ≠ Friday traffic) — adaptive 3σ detection |
| **Krum aggregation** | Mathematically proven Byzantine-fault tolerance — filters poisoned model updates |
| **IsolationForest sanitisation** | Removes 2% extreme outliers from benign baseline before scaler fitting — prevents data poisoning |
| **Never isolate critical infra** | Auto-isolating a Domain Controller is worse than the attack — HITL required |
| **Blockchain for model hashes** | Supply-chain attack defence — detects silent model weight tampering |

---

## Attack Simulation (Dashboard)

The dashboard includes 5 red-team attack profiles:

| Attack | Feature Perturbation |
|---|---|
| **DDoS Flood** | Max packet rate, near-zero IAT, high SYN flags |
| **Port Scan** | Near-zero duration/bytes, high RST flags |
| **Lateral Movement** | High IAT std (beaconing), edges rewired to critical nodes |
| **Data Exfiltration** | Very high forward bytes, near-zero backward bytes |
| **Web Attack** | Large payload, high PSH flags, short duration |

---

## Response Policy Matrix

| Severity | Node Type | Action | Command |
|---|---|---|---|
| LOW | Any | `LOG_ONLY` | — |
| MEDIUM | Any | `THROTTLE` + HITL | `tc qdisc htb rate 10kbps` |
| HIGH | Critical (DC / SCADA / DB / Payment GW) | `THROTTLE` + HITL | Never auto-isolate |
| HIGH | Standard workstation | `ISOLATE` | `iptables -A INPUT -s <IP> -j DROP` |

---

## Model Parameters

| Component | Architecture | Parameters |
|---|---|---|
| FlowAutoencoder | 78→64→32→16→32→64→78 | 15,390 |
| AuraSTGNN | SAGEConv 78→64→32→1 | 14,913 |
| **Total** | | **30,303** |

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | PyTorch 2.10 (CPU) |
| GNN | Manual GraphSAGE (`torch.scatter_add_`) |
| Federated Learning | Flower (`flwr`) |
| Anomaly Detection | scikit-learn IsolationForest + custom EMA |
| Blockchain | Web3.py + Solidity 0.8.19 / local fallback |
| Dashboard | Streamlit + Plotly |
| Data | pandas, numpy, networkx |

---

## Team

**Team Trinetra** · NEXJEM Hackathon 2026
