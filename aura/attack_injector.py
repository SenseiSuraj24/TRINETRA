"""
aura/attack_injector.py — Live Attack Simulation Engine
========================================================

This module is the "Red Button" for the hackathon demo.

It generates synthetic malicious network flow vectors that mimic
real-world attack signatures from the CICIDS2017 taxonomy:

  1. DDoS (Distributed Denial of Service)
     → Flood packet rates, low IAT, uniform packet sizes
  2. Port Scan
     → Many distinct destination ports, low byte count, high SYN flags
  3. Lateral Movement
     → Unusual src→dst connectivity, internal IP-space traversal
  4. Data Exfiltration (Low & Slow)
     → Abnormal byte ratio, robotic jitter timing, sustained outbound flow
  5. Web Attack (SQL Injection fingerprint)
     → Short flows, high PSH flags, response anomalies

Each attack type perturbs specific feature dimensions (matching the
CICIDS2017 column schema) to create realistic anomaly signatures that
the autoencoder WILL flag — because they deviate from the normal manifold.

Usage
-----
>>> injector = AttackInjector()
>>> attack_graph = injector.inject("ddos", graph_dict)
>>> # Feed attack_graph to AURAInferenceEngine — watch the alert fire
"""

import logging
import time
from enum import Enum
from typing import Dict, Optional

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Attack Type Registry
# ─────────────────────────────────────────────────────────────────────────────

class AttackType(Enum):
    DDOS             = "DDoS"
    PORT_SCAN        = "PortScan"
    LATERAL_MOVEMENT = "LateralMovement"
    EXFILTRATION     = "Exfiltration"
    WEB_ATTACK       = "WebAttack"


# CICIDS2017 feature index mapping (from inspect_csv.py output)
# These are the normalised [0-1] indices we can manipulate for realistic attacks.
FEATURE_MAP = {
    "dest_port":          0,   # Destination Port
    "flow_duration":      1,   # Flow Duration
    "fwd_packets":        2,   # Total Fwd Packets
    "bwd_packets":        3,   # Total Backward Packets
    "fwd_bytes":          4,   # Total Length of Fwd Packets
    "bwd_bytes":          5,   # Total Length of Bwd Packets
    "fwd_pkt_len_max":    6,
    "fwd_pkt_len_min":    7,
    "flow_bytes_s":       14,  # Flow Bytes/s
    "flow_pkts_s":        15,  # Flow Packets/s
    "flow_iat_mean":      16,  # Flow Inter-Arrival Time mean
    "flow_iat_std":       17,
    "fwd_iat_total":      20,
    "fwd_psh_flags":      30,  # Forward PSH flags
    "syn_flag_count":     44,  # SYN Flag Count
    "rst_flag_count":     45,
    "psh_flag_count":     46,
    "ack_flag_count":     47,
    "subflow_fwd_bytes":  63,
    "subflow_bwd_bytes":  65,
    "idle_mean":          74,  # Idle Mean
    "idle_std":           75,
}


# ─────────────────────────────────────────────────────────────────────────────
# Attack Signature Profiles
# ─────────────────────────────────────────────────────────────────────────────

def _ddos_profile(n: int, f: int) -> np.ndarray:
    """
    DDoS Attack Signature
    ---------------------
    Characteristics: UDP/ICMP flood, extremely high packet rate, low IAT,
    uniform packet sizes, no TCP handshake (SYN without ACK).

    Feature perturbations:
    - flow_pkts_s: → 0.95+ (near-max packet rate)
    - flow_bytes_s: → 0.90+ (high bandwidth utilisation)
    - flow_iat_mean: → 0.02 (near-zero inter-arrival time = flood)
    - flow_iat_std: → 0.01 (robotic regularity)
    - syn_flag_count: → 0.85 (mass SYN without corresponding ACK)
    - ack_flag_count: → 0.05 (very few ACKs = incomplete handshakes)
    - fwd_packets: → 0.95 (overwhelming forward packet count)
    """
    x = np.random.uniform(0.3, 0.6, (n, f)).astype(np.float32)  # baseline normal
    x[:, FEATURE_MAP["flow_pkts_s"]]    = np.random.uniform(0.90, 0.99, n)
    x[:, FEATURE_MAP["flow_bytes_s"]]   = np.random.uniform(0.88, 0.99, n)
    x[:, FEATURE_MAP["flow_iat_mean"]]  = np.random.uniform(0.00, 0.03, n)
    x[:, FEATURE_MAP["flow_iat_std"]]   = np.random.uniform(0.00, 0.02, n)
    x[:, FEATURE_MAP["syn_flag_count"]] = np.random.uniform(0.80, 0.99, n)
    x[:, FEATURE_MAP["ack_flag_count"]] = np.random.uniform(0.01, 0.08, n)
    x[:, FEATURE_MAP["fwd_packets"]]    = np.random.uniform(0.90, 0.99, n)
    return x


def _portscan_profile(n: int, f: int) -> np.ndarray:
    """
    Port Scan Signature
    -------------------
    Characteristics: Very short flows to many destination ports,
    minimal data transferred, high RST flag rate (closed ports).

    Feature perturbations:
    - flow_duration: → 0.02 (extremely short flows)
    - fwd_bytes: → 0.03 (minimal data, just probe packets)
    - bwd_bytes: → 0.02 (minimal response)
    - rst_flag_count: → 0.88 (RST = port closed/filtered)
    - syn_flag_count: → 0.75 (SYN probe without completing handshake)
    - flow_bytes_s: → 0.15 (low throughput)
    """
    x = np.random.uniform(0.3, 0.6, (n, f)).astype(np.float32)
    x[:, FEATURE_MAP["flow_duration"]]  = np.random.uniform(0.00, 0.03, n)
    x[:, FEATURE_MAP["fwd_bytes"]]      = np.random.uniform(0.00, 0.04, n)
    x[:, FEATURE_MAP["bwd_bytes"]]      = np.random.uniform(0.00, 0.03, n)
    x[:, FEATURE_MAP["rst_flag_count"]] = np.random.uniform(0.80, 0.99, n)
    x[:, FEATURE_MAP["syn_flag_count"]] = np.random.uniform(0.70, 0.90, n)
    x[:, FEATURE_MAP["flow_bytes_s"]]   = np.random.uniform(0.05, 0.15, n)
    return x


def _lateral_movement_profile(n: int, f: int) -> np.ndarray:
    """
    Lateral Movement Signature
    --------------------------
    Characteristics: Legitimate-looking TCP flows but to unusual internal
    destinations, moderate data transfer, abnormal timing jitter.
    The GNN (Layer 2) is SPECIFICALLY designed to catch this via topological
    anomaly detection — IP A should not be talking to Database Server B.

    Feature perturbations:
    - flow_duration: → 0.6 (medium duration)
    - fwd_packets: → 0.55 (moderate traffic)
    - flow_iat_std: → 0.80 (high timing jitter = evasion attempt)
    - idle_mean: → 0.85 (long idle periods between bursts = reconnaissance)
    - psh_flag_count: → 0.70 (data transfer)
    """
    x = np.random.uniform(0.3, 0.6, (n, f)).astype(np.float32)
    x[:, FEATURE_MAP["flow_duration"]]  = np.random.uniform(0.50, 0.75, n)
    x[:, FEATURE_MAP["fwd_packets"]]    = np.random.uniform(0.50, 0.65, n)
    x[:, FEATURE_MAP["flow_iat_std"]]   = np.random.uniform(0.75, 0.95, n)   # Jitter
    x[:, FEATURE_MAP["idle_mean"]]      = np.random.uniform(0.80, 0.98, n)   # Beacon-like
    x[:, FEATURE_MAP["idle_std"]]       = np.random.uniform(0.03, 0.10, n)   # Robotic regularity
    x[:, FEATURE_MAP["psh_flag_count"]] = np.random.uniform(0.60, 0.80, n)
    return x


def _exfiltration_profile(n: int, f: int) -> np.ndarray:
    """
    Data Exfiltration (Low & Slow) Signature
    -----------------------------------------
    Characteristics: Sustained outbound flow with high forward-to-backward
    byte ratio (data leaving, no data coming back), robotic inter-arrival
    timing (scripted exfil), unusual flow duration.

    Feature perturbations:
    - fwd_bytes: → 0.92 (large outbound data)
    - bwd_bytes: → 0.05 (nothing coming back)
    - subflow_fwd_bytes: → 0.90
    - flow_iat_mean: → 0.45 (regulated pacing to evade rate limits)
    - flow_iat_std: → 0.02 (robotic = machine-generated timing)
    - flow_duration: → 0.88 (very long, sustained connection)
    """
    x = np.random.uniform(0.3, 0.6, (n, f)).astype(np.float32)
    x[:, FEATURE_MAP["fwd_bytes"]]         = np.random.uniform(0.88, 0.99, n)
    x[:, FEATURE_MAP["bwd_bytes"]]         = np.random.uniform(0.00, 0.06, n)
    x[:, FEATURE_MAP["subflow_fwd_bytes"]] = np.random.uniform(0.85, 0.99, n)
    x[:, FEATURE_MAP["subflow_bwd_bytes"]] = np.random.uniform(0.00, 0.04, n)
    x[:, FEATURE_MAP["flow_iat_mean"]]     = np.random.uniform(0.40, 0.55, n)
    x[:, FEATURE_MAP["flow_iat_std"]]      = np.random.uniform(0.00, 0.03, n)
    x[:, FEATURE_MAP["flow_duration"]]     = np.random.uniform(0.80, 0.98, n)
    return x


def _web_attack_profile(n: int, f: int) -> np.ndarray:
    """
    Web Attack (SQL Injection / XSS) Signature
    -------------------------------------------
    Characteristics: Abnormally short requests with high PSH flag counts,
    unusual response sizes (error pages vs normal content), low IAT between
    repeated injection attempts.

    Feature perturbations:
    - fwd_bytes: → 0.85 (large request payload: injected SQL string)
    - bwd_bytes: → 0.25 (error/response page)
    - psh_flag_count: → 0.90 (immediate data push)
    - flow_duration: → 0.08 (short burst)
    - flow_iat_mean: → 0.05 (rapid repeated requests)
    """
    x = np.random.uniform(0.3, 0.6, (n, f)).astype(np.float32)
    x[:, FEATURE_MAP["fwd_bytes"]]      = np.random.uniform(0.80, 0.95, n)
    x[:, FEATURE_MAP["bwd_bytes"]]      = np.random.uniform(0.20, 0.30, n)
    x[:, FEATURE_MAP["psh_flag_count"]] = np.random.uniform(0.85, 0.98, n)
    x[:, FEATURE_MAP["flow_duration"]]  = np.random.uniform(0.05, 0.12, n)
    x[:, FEATURE_MAP["flow_iat_mean"]]  = np.random.uniform(0.02, 0.08, n)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Attack Injector
# ─────────────────────────────────────────────────────────────────────────────

ATTACK_PROFILES = {
    AttackType.DDOS:             _ddos_profile,
    AttackType.PORT_SCAN:        _portscan_profile,
    AttackType.LATERAL_MOVEMENT: _lateral_movement_profile,
    AttackType.EXFILTRATION:     _exfiltration_profile,
    AttackType.WEB_ATTACK:       _web_attack_profile,
}


class AttackInjector:
    """
    Generates synthetic attack graph snapshots for demo injection.

    Either creates a fresh graph from scratch (standalone demo) or
    overwrites edge_attr in an existing graph_dict (live injection).
    """

    def __init__(
        self,
        num_nodes:   int = cfg.NUM_SYNTHETIC_NODES,
        feature_dim: int = cfg.FEATURE_DIM,
        num_edges:   int = 40,
    ):
        self.num_nodes   = num_nodes
        self.feature_dim = feature_dim
        self.num_edges   = num_edges

    def inject(
        self,
        attack_type: str,                           # "ddos", "portscan", etc.
        base_graph:  Optional[Dict] = None,         # If provided, mutate in place
        n_attacked_edges: Optional[int] = None,     # How many edges to infect
    ) -> Dict:
        """
        Inject attack traffic into a graph snapshot.

        If base_graph is None, generates a fresh healthy graph first,
        then corrupts n_attacked_edges of it with attack-pattern features.

        Parameters
        ----------
        attack_type       : One of "ddos", "portscan", "lateral", "exfil", "web"
        base_graph        : Existing graph_dict to inject into (optional)
        n_attacked_edges  : Number of edges to corrupt (default = 30% of edges)

        Returns
        -------
        Mutated graph_dict with attack traces in edge_attr + metadata
        """
        attack_enum = self._resolve_attack_type(attack_type)
        profile_fn  = ATTACK_PROFILES[attack_enum]

        if base_graph is None:
            base_graph = self._generate_healthy_graph()

        edge_attr  = base_graph["edge_attr"].clone()          # [E, F]
        n_edges    = edge_attr.shape[0]
        n_attacked = n_attacked_edges or max(1, int(n_edges * 0.30))
        n_attacked = min(n_attacked, n_edges)

        # Choose which edges to infect (first n_attacked for determinism in demo)
        attack_features = profile_fn(n_attacked, self.feature_dim)
        edge_attr[:n_attacked] = torch.tensor(attack_features, dtype=torch.float32)

        # For lateral movement: rewire some edges to anomalous topology
        edge_index = base_graph["edge_index"].clone()
        if attack_enum == AttackType.LATERAL_MOVEMENT:
            edge_index = self._rewire_edges(edge_index, n_attacked)

        # Recompute node features from updated edges
        from aura.data_loader import _build_node_features
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        X_node = _build_node_features(
            edge_attr.numpy(), src, dst, self.num_nodes, self.feature_dim
        )

        attack_graph = {
            "x":           torch.tensor(X_node, dtype=torch.float32),
            "edge_index":  edge_index,
            "edge_attr":   edge_attr,
            "ttl_state":   base_graph.get("ttl_state", {}),
            "window_id":   f"INJECTED_{attack_enum.value}_{int(time.time())}",
            "attack_type": attack_enum.value,
            "attack_nodes": list(range(min(5, self.num_nodes))),  # First few nodes compromised
            "n_attacked_edges": n_attacked,
        }

        logger.info(
            f"[INJECTOR] {attack_enum.value} attack injected into "
            f"{n_attacked}/{n_edges} edges."
        )
        return attack_graph

    def generate_attack_stream(self, attack_type: str, n_windows: int = 5):
        """
        Generator that yields n_windows consecutive attack graph snapshots.
        Used for sustained attack simulation in the dashboard demo.
        """
        for i in range(n_windows):
            yield self.inject(attack_type), i

    def _generate_healthy_graph(self) -> Dict:
        """Generate a baseline healthy network graph."""
        # Normal traffic: Gaussian centred at ~0.4 in normalised space
        edge_attr  = torch.rand(self.num_edges, self.feature_dim) * 0.3 + 0.3
        edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))

        # Remove self-loops
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        edge_attr  = edge_attr[:mask.sum()]

        from aura.data_loader import _build_node_features
        X_node = _build_node_features(
            edge_attr.numpy(),
            edge_index[0].numpy(),
            edge_index[1].numpy(),
            self.num_nodes,
            self.feature_dim,
        )
        return {
            "x":          torch.tensor(X_node, dtype=torch.float32),
            "edge_index": edge_index,
            "edge_attr":  edge_attr,
            "ttl_state":  {},
            "window_id":  f"HEALTHY_{int(time.time())}",
        }

    def _rewire_edges(
        self, edge_index: torch.Tensor, n_rewire: int
    ) -> torch.Tensor:
        """
        For lateral movement simulation: force the first n_rewire edges to
        connect 'workstation' nodes (high IDs) to 'server' nodes (low IDs).
        This creates topologically anomalous connections the GNN can detect.
        """
        ei = edge_index.clone()
        n  = self.num_nodes
        for i in range(min(n_rewire, ei.shape[1])):
            # Src: random workstation (last 1/3 of nodes)
            ei[0, i] = np.random.randint(n * 2 // 3, n)
            # Dst: critical server (first 4 nodes = the CRITICAL_ALLOWLIST nodes)
            ei[1, i] = np.random.randint(0, 4)
        return ei

    @staticmethod
    def _resolve_attack_type(s: str) -> AttackType:
        """Map user-friendly string to AttackType enum."""
        mapping = {
            "ddos":     AttackType.DDOS,
            "portscan": AttackType.PORT_SCAN,
            "port_scan": AttackType.PORT_SCAN,
            "lateral":  AttackType.LATERAL_MOVEMENT,
            "lateral_movement": AttackType.LATERAL_MOVEMENT,
            "exfil":    AttackType.EXFILTRATION,
            "exfiltration": AttackType.EXFILTRATION,
            "web":      AttackType.WEB_ATTACK,
            "web_attack": AttackType.WEB_ATTACK,
        }
        key = s.lower().replace("-", "_")
        if key not in mapping:
            raise ValueError(f"Unknown attack type '{s}'. Choose: {list(mapping.keys())}")
        return mapping[key]


# ─────────────────────────────────────────────────────────────────────────────
# CLI Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== AURA Attack Injector — Demo ===\n")
    injector = AttackInjector()

    for attack in ["ddos", "portscan", "lateral", "exfil", "web"]:
        g = injector.inject(attack)
        ea = g["edge_attr"]
        print(f"  {g['attack_type']:20s} | edges={ea.shape[0]}  "
              f"mean={ea.mean():.4f}  max={ea.max():.4f}  "
              f"attacked={g['n_attacked_edges']}")

    print("\n✓ Attack injector test passed.")
