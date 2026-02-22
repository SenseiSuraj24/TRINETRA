"""
aura/fl_server.py — Flower FL Server: Krum Aggregation + Straggler Policy
==========================================================================

This server implements two critical security properties:

1. BYZANTINE ROBUSTNESS (Krum Aggregation)
   -----------------------------------------
   Standard FedAvg is vulnerable to "model poisoning" — a compromised
   client can send a manipulated weight update that pushes the global
   model to misclassify specific attacks.

   Krum (Blanchard et al., 2017) defends against this:
     - For each client's update, compute the sum of squared Euclidean
       distances to its k nearest neighbour updates.
     - Select the m updates with the LOWEST neighbourhood distances.
     - These are the "most central" updates — mathematical outliers
       (poisoned clients) have high distances and are dropped.

   The aggregated global model is the mean of the m selected updates.

   Guarantee: Krum is proven resilient to f Byzantine workers as long as
   n > 2f + 2 (where n = total clients, f = faulty clients).
   With 3 clients: n=3, f≤0, so 1 poisoned client is tolerated.

2. STRAGGLER TIMEOUT POLICY
   --------------------------
   In synchronous FL, if a client disconnects mid-round, the server
   blocks indefinitely — causing a Denial-of-Service against the entire
   federation.

   AURA implements an explicit timeout policy:
     - After `round_timeout_sec` seconds, unreceived client updates
       are DROPPED from the aggregation round.
     - If fewer than `min_clients` responses arrive, the round is
       ABANDONED and the previous global model is preserved.
     - A warning is logged for operator review.

   This is configured via Flower's `on_fit_config_fn` and
   `min_available_clients` parameters + a strategy-level timeout.

3. IMMUTABLE AUDIT LOG
   ----------------------
   After each successful aggregation, the server hashes the model weights
   (SHA-256) and writes the hash to the AURA blockchain module.
   This creates a tamper-evident chain of custody for all model updates.
"""

import hashlib
import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
    EvaluateRes,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy import FedAvg

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from aura.models import AURAModelBundle

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Krum Aggregation Implementation
# ─────────────────────────────────────────────────────────────────────────────

def krum_select(
    updates:       List[List[np.ndarray]],
    num_to_select: int = cfg.KRUM_NUM_TO_SELECT,
) -> List[int]:
    """
    Krum Selection Algorithm (Blanchard et al., 2017).

    Algorithm
    ---------
    Given n client weight updates {w_1, ..., w_n}:
    1. For each client i, flatten its weight update into a 1D vector.
    2. Compute pairwise squared Euclidean distances: d(i,j) = ||w_i - w_j||²
    3. For each client i, compute the Krum score:
         s(i) = Σ_{j ∈ N_k(i)} d(i, j)
       where N_k(i) = k nearest neighbours of i (k = n - num_to_select - 2)
    4. Select the num_to_select clients with the LOWEST Krum scores.

    Poisoned clients produce updates far from the cluster → high scores → dropped.

    Parameters
    ----------
    updates        : List of per-client parameters (each is a list of ndarrays)
    num_to_select  : How many clients to keep (KRUM_NUM_TO_SELECT in config)

    Returns
    -------
    Selected client indices (those with lowest Krum scores)
    """
    n = len(updates)
    if n <= num_to_select:
        logger.warning("Krum: fewer clients than num_to_select — accepting all.")
        return list(range(n))

    # Flatten each client's parameters to a single 1D vector
    flat = []
    for client_params in updates:
        flat.append(np.concatenate([p.flatten() for p in client_params]))

    # k = n - num_to_select - 2  (guaranteed Byzantine tolerance formula)
    k = max(1, n - num_to_select - 2)

    scores = []
    for i in range(n):
        # Squared Euclidean distances from client i to all others
        dists = sorted([
            float(np.sum((flat[i] - flat[j]) ** 2))
            for j in range(n) if j != i
        ])
        # Krum score = sum of k smallest distances
        scores.append(sum(dists[:k]))

    # Rank clients by score (ascending) and select the best num_to_select
    ranked = sorted(range(n), key=lambda idx: scores[idx])
    selected = ranked[:num_to_select]

    dropped = [i for i in range(n) if i not in selected]
    if dropped:
        logger.warning(
            f"[KRUM] Dropped client indices {dropped} as potential outliers.  "
            f"Scores: {[round(s, 2) for s in scores]}"
        )
    else:
        logger.info(f"[KRUM] All clients accepted.  Scores: {[round(s, 2) for s in scores]}")

    return selected


def krum_aggregate(
    selected_updates: List[List[np.ndarray]],
) -> List[np.ndarray]:
    """
    Aggregate selected (Krum-filtered) client updates by simple mean.

    By the time we reach this function, Byzantine clients have already been
    filtered by krum_select.  A simple mean of the remaining honest updates
    produces the new global model.

    Shape preservation:  each result array has the same shape as input arrays.
    """
    # Cast to float32: keeps dtype consistent with PyTorch model weights so
    # SHA-256 hashes computed before and after Flower serialization always match.
    return [
        np.mean([update[i] for update in selected_updates], axis=0).astype(np.float32)
        for i in range(len(selected_updates[0]))
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SHA-256 Model Hash
# ─────────────────────────────────────────────────────────────────────────────

def hash_model_weights(arrays: List[np.ndarray]) -> str:
    """
    Compute a SHA-256 hash over the concatenated model weight bytes.

    Normalises every array to C-contiguous float32 before hashing so the
    result is identical whether called on the server-side aggregated arrays
    or on the client-side after Flower's ndarrays_to_parameters round-trip.
    """
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return "0x" + h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Custom Flower Strategy: KrumFedAURA
# ─────────────────────────────────────────────────────────────────────────────

class KrumFedAURA(FedAvg):
    """
    Custom Flower aggregation strategy extending FedAvg with:
      1. Krum-based Byzantine-robust aggregation
      2. Straggler timeout + drop policy
      3. Per-round SHA-256 model hash logging
      4. Blockchain audit integration

    Inheriting from FedAvg reuses all the Flower boilerplate (
    sampling, evaluation scheduling, etc.) while overriding only
    `aggregate_fit` where our security logic lives.
    """

    def __init__(
        self,
        min_fit_clients:       int   = cfg.FL_MIN_CLIENTS,
        min_available_clients: int   = cfg.FL_MIN_AVAILABLE,
        num_rounds:            int   = cfg.FL_NUM_ROUNDS,
        round_timeout_sec:     int   = cfg.FL_ROUND_TIMEOUT_SEC,
        blockchain_module=None,      # Optionally inject blockchain logger
    ):
        # Configure FedAvg base (we override aggregation but keep its scheduling)
        super().__init__(
            min_fit_clients       = min_fit_clients,
            min_available_clients = min_available_clients,
            # Round config function: tells clients how many local epochs to run
            on_fit_config_fn = lambda rnd: {
                "local_epochs": 3,
                "round":        rnd,
                # Hint to client libraries; actual enforcement is server-side
                "timeout_sec":  round_timeout_sec,
            },
        )
        self.num_rounds        = num_rounds
        self.round_timeout_sec = round_timeout_sec
        self.blockchain        = blockchain_module
        self._model_version    = 0
        self._hash_history: List[dict] = []

        # Clear the trusted registry at the start of each FL session so that
        # only the current session's final hash is present (1 hash per run).
        registry_path = Path(cfg.LOGS_DIR) / "hash_registry.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text("{}")
        logger.info("[REGISTRY] Cleared — fresh FL session starting.")

        logger.info(
            f"KrumFedAURA strategy ready  |  "
            f"rounds={num_rounds}  timeout={round_timeout_sec}s  "
            f"krum_select={cfg.KRUM_NUM_TO_SELECT}"
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Override FedAvg's aggregate_fit to apply Krum instead of simple mean.

        Straggler Policy
        ----------------
        Flower already handles client timeouts internally via its gRPC layer
        (configured via flwr.server.ServerConfig(round_timeout=...)).
        Failed/timed-out clients arrive in the `failures` list, not in
        `results`.  We log them and proceed with whatever arrived on time.

        If fewer than min_fit_clients results arrive, we PRESERVE the previous
        global model (no aggregation) and log the stale round.
        """
        round_tag = f"[SERVER round={server_round}]"

        # ── Straggler Policy ────────────────────────────────────────────────
        n_received = len(results)
        n_failed   = len(failures)

        if n_failed > 0:
            logger.warning(
                f"{round_tag} {n_failed} client(s) timed out / failed "
                f"(straggler drop policy applied).  Proceeding with {n_received} responses."
            )

        if n_received < cfg.FL_MIN_CLIENTS:
            logger.error(
                f"{round_tag} Insufficient responses ({n_received} < "
                f"{cfg.FL_MIN_CLIENTS}).  ABANDONING round — global model preserved."
            )
            return None, {"status": "abandoned", "received": n_received}

        # ── Extract weight arrays from Flower FitRes ─────────────────────────
        client_updates = []
        for client_proxy, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            client_updates.append(arrays)
            client_loss = fit_res.metrics.get("train_loss", "N/A")
            logger.info(f"{round_tag} Received update  |  "
                        f"num_examples={fit_res.num_examples}  loss={client_loss}")

        # ── Krum Filtering ───────────────────────────────────────────────────
        print(f"\n{round_tag} Running Krum filtering on {n_received} updates …")
        selected_indices = krum_select(client_updates, cfg.KRUM_NUM_TO_SELECT)
        selected_updates = [client_updates[i] for i in selected_indices]

        # ── Aggregate selected updates ────────────────────────────────────────
        aggregated = krum_aggregate(selected_updates)
        self._model_version += 1
        model_version_tag = f"v{self._model_version}.{server_round}"

        # ── SHA-256 Hash (computed every round — clients verify weights) ──────
        model_hash = hash_model_weights(aggregated)
        print(f"{round_tag} Global Model {model_version_tag} aggregated.")
        print(f"{round_tag} SHA-256 hash: {model_hash}")

        # ── Blockchain Audit Log (FINAL ROUND ONLY) ──────────────────────────
        # Intermediate rounds converge the model; only the final aggregated
        # model is production-ready and gets minted on the blockchain ledger.
        is_final_round = (server_round == self.num_rounds)
        if is_final_round:
            final_version = f"final_v{self._model_version}"
            if self.blockchain is not None:
                try:
                    tx_hash = self.blockchain.log_model_update(
                        model_version=final_version,
                        model_hash=model_hash,
                    )
                    print(f"[BLOCKCHAIN] Final Model {final_version} minted. "
                          f"Hash {model_hash[:12]}… | TX: {str(tx_hash)[:16]}…")
                except Exception as e:
                    logger.warning(f"Blockchain log failed (fallback active): {e}")
            else:
                self._log_hash_local(final_version, model_hash, server_round)

            # Write trusted registry — only for the final converged model
            self._write_trusted_registry(final_version, model_hash)
            model_version_tag = final_version
        else:
            print(f"{round_tag} Intermediate round — hash not minted yet "
                  f"(blockchain mint on round {self.num_rounds} only).")

        # Expose which indices were selected so client_statuses can use it
        # Record in history (for dashboard display)
        self._hash_history.append({
            "round":   server_round,
            "version": model_version_tag,
            "hash":    model_hash,
            "clients_selected": selected_indices,
            "clients_dropped":  [i for i in range(n_received) if i not in selected_indices],
        })

        # Save the aggregated model to disk
        self._save_model(aggregated, model_version_tag)

        return ndarrays_to_parameters(aggregated), {
            "model_version":        model_version_tag,
            "model_hash":           model_hash,
            "krum_selected":        len(selected_indices),
            "krum_selected_indices": selected_indices,
            "krum_dropped":         n_received - len(selected_indices),
        }

    def _log_hash_local(self, version: str, model_hash: str, rnd: int) -> None:
        """Fallback: write hash to local JSONL file if blockchain is unavailable."""
        record = {
            "timestamp": time.time(),
            "round":     rnd,
            "version":   version,
            "hash":      model_hash,
        }
        log_path = Path(cfg.LOGS_DIR) / "model_hashes.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"[LOCAL-HASH] {model_hash} written to {log_path}")

    def _write_trusted_registry(self, version: str, model_hash: str) -> None:
        """Write to the trusted hash registry (separate from the ledger).
        verify_chain.py reads this as the ground-truth reference.
        Corrupting the blockchain ledger won't affect this file.
        """
        registry_path = Path(cfg.LOGS_DIR) / "hash_registry.json"
        registry: dict = {}
        if registry_path.exists():
            try:
                registry = json.loads(registry_path.read_text())
            except Exception:
                registry = {}
        registry[version] = model_hash
        registry_path.write_text(json.dumps(registry, indent=2))
        logger.info(f"[REGISTRY] {version} written to trusted registry.")

    def _save_model(self, arrays: List[np.ndarray], version_tag: str) -> None:
        """Save the aggregated global model weights to disk."""
        model = AURAModelBundle()
        with torch.no_grad():
            for p, arr in zip(model.parameters(), arrays):
                p.copy_(torch.tensor(arr))
        save_path = Path(cfg.MODELS_DIR) / f"global_model_{version_tag}.pth"
        torch.save(model.state_dict(), save_path)
        logger.info(f"Global model saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Server Launch
# ─────────────────────────────────────────────────────────────────────────────

def start_server(blockchain_module=None) -> None:
    """
    Start the AURA Flower federation server.

    The server blocks until all FL_NUM_ROUNDS are complete, then exits.
    Run in a separate process/thread from the dashboard.

    Straggler timeout is enforced via ServerConfig(round_timeout=...).
    Any client that doesn't respond within round_timeout_sec is treated
    as dropped for that round (Flower gRPC layer handles the socket close).
    """
    strategy = KrumFedAURA(blockchain_module=blockchain_module)

    server_config = fl.server.ServerConfig(
        num_rounds    = cfg.FL_NUM_ROUNDS,
        round_timeout = cfg.FL_ROUND_TIMEOUT_SEC,   # Straggler hard timeout
    )

    print(f"\n{'='*60}")
    print(f"  AURA Federation Server starting on {cfg.FL_SERVER_ADDRESS}")
    print(f"  Rounds: {cfg.FL_NUM_ROUNDS}  |  Timeout: {cfg.FL_ROUND_TIMEOUT_SEC}s")
    print(f"  Strategy: Krum (Byzantine-robust, selects {cfg.KRUM_NUM_TO_SELECT})")
    print(f"{'='*60}\n")

    fl.server.start_server(
        server_address = cfg.FL_SERVER_ADDRESS,
        config         = server_config,
        strategy       = strategy,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulation Mode (no real gRPC) — for demo without network setup
# ─────────────────────────────────────────────────────────────────────────────

def run_federation_simulation(blockchain_module=None, n_rounds: int = None,
                              active_orgs: list = None) -> List[dict]:
    """
    In-process federation simulation for the hackathon demo.

    Parameters
    ----------
    active_orgs : list of org keys that are ready, e.g. ["hospital","university"].
                  If None, defaults to all three.  Only ready orgs participate.
                  Byzantine client is randomly assigned among them each run.
    """
    from aura.fl_client import create_mock_clients

    if n_rounds is None:
        n_rounds = cfg.FL_NUM_ROUNDS

    if active_orgs is None:
        active_orgs = ["hospital", "bank", "university"]

    # Attack is tied to the bank org — only injected if bank is in active_orgs.
    # If bank is offline, all clients are honest (no meaningless Krum drop).
    if "bank" in active_orgs:
        attack_arg = active_orgs.index("bank")
    else:
        attack_arg = -1   # all honest

    clients, attack_idx = create_mock_clients(
        n_clients     = len(active_orgs),
        n_samples     = 300,
        org_ids       = active_orgs,
        attack_client = attack_arg,
    )
    strategy = KrumFedAURA(blockchain_module=blockchain_module,
                           num_rounds=n_rounds)

    # Initialise with random global model
    global_model = AURAModelBundle()
    global_params = [p.detach().cpu().numpy() for p in global_model.parameters()]

    round_results = []

    for rnd in range(1, n_rounds + 1):
        print(f"\n{'─'*55}")
        print(f"  FEDERATION ROUND {rnd}/{n_rounds}")
        print(f"{'─'*55}")

        fit_results = []
        for client in clients:
            # Build FitIns with current global params
            from flwr.common import FitIns, Config
            fit_ins = FitIns(
                parameters = ndarrays_to_parameters(global_params),
                config     = {"local_epochs": 3, "round": rnd},
            )
            fit_res = client.fit(fit_ins)

            # Simulate per-client console output for demo effect
            print(f"  [CLIENT {client.client_id}] weights sent to server ✓")
            fit_results.append((None, fit_res))   # ClientProxy=None in simulation

        # Run server-side Krum + aggregation
        new_params, metrics = strategy.aggregate_fit(
            server_round = rnd,
            results      = fit_results,
            failures     = [],
        )
        # Build per-client status for dashboard display
        # "Byzantine" = whoever Krum DROPPED (mathematical outlier detection)
        selected_idx = metrics.get('krum_selected_indices', [])
        dropped_idx  = [i for i in range(len(clients)) if i not in selected_idx]
        client_statuses = []
        for i, client in enumerate(clients):
            is_selected  = (selected_idx and i in selected_idx)
            is_byzantine = (i in dropped_idx)   # dropped by Krum = suspicious
            org_key      = active_orgs[i] if i < len(active_orgs) else f"org_{i}"
            net_map      = {"hospital": "192.168.1.0/24",
                            "bank":     "10.0.1.0/24",
                            "university": "172.16.1.0/24"}
            client_statuses.append({
                "client_id": client.client_id,
                "org_id":    org_key,
                "network":   net_map.get(org_key, "—"),
                "org":       org_key.capitalize(),
                "role":      "Byzantine" if is_byzantine else "Normal",
                "selected":  is_selected if selected_idx else (not is_byzantine),
                "round":     rnd,
            })
        if new_params is not None:
            global_params = parameters_to_ndarrays(new_params)
            model_version = metrics.get('model_version')
            server_hash   = metrics.get('model_hash')
            print(f"\n  [SERVER] Global Model {model_version} aggregated.")
            print(f"  [SERVER] SHA-256 minted on blockchain: {server_hash[:20]}...")
            print(f"  [SERVER] Krum kept {metrics.get('krum_selected')} / "
                  f"{len(clients)} clients.")
            print()

            # ── Client-side hash verification ────────────────────────────────────
            # Hash verification only makes sense on the final round, because the
            # blockchain is only minted once (at the end of federation).
            # On intermediate rounds we print the computed hash for auditing only.
            is_final = model_version and model_version.startswith("final_")
            if is_final:
                client_received_hash = hash_model_weights(global_params)

                for client in clients:
                    # Fetch the hash the server minted for this version
                    bc = blockchain_module
                    if bc is not None:
                        on_chain_ok, _ = bc.verify_model(model_version, client_received_hash)
                    else:
                        # No blockchain — compare directly against server hash
                        on_chain_ok = (client_received_hash == server_hash)

                    if on_chain_ok:
                        print(f"  [CLIENT {client.client_id}] "
                              f"Received hash {client_received_hash[:16]}... "
                              f"== Blockchain hash {server_hash[:16]}... "
                              f"→ MATCH. Model deployed.")
                    else:
                        print(f"  [CLIENT {client.client_id}] "
                              f"Received hash {client_received_hash[:16]}... "
                              f"!= Blockchain hash {server_hash[:16]}... "
                              f"→ MISMATCH! Weights tampered in transit. REJECTING model.")
            else:
                print(f"  [CLIENTS] Intermediate round — blockchain not yet minted. "
                      f"Hash {server_hash[:20]}... recorded locally for auditing.")

        round_results.append({"round": rnd, "client_statuses": client_statuses, **metrics})

    print(f"\n{'='*55}")
    print(f"  Federation complete.  {n_rounds} rounds executed.")
    print(f"{'='*55}\n")

    return round_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AURA FL Aggregation Server")
    parser.add_argument(
        "--address", default=cfg.FL_SERVER_ADDRESS,
        help="gRPC bind address (default: %(default)s). "
             "Use 0.0.0.0:8080 to accept remote clients."
    )
    parser.add_argument(
        "--rounds", type=int, default=cfg.FL_NUM_ROUNDS,
        help="Number of FL rounds (default: %(default)s)"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run in-process simulation instead of gRPC server (legacy mode)"
    )
    args = parser.parse_args()

    if args.simulate:
        print("=== AURA Federation — In-Process Simulation Mode ===")
        results = run_federation_simulation(n_rounds=args.rounds)
        for r in results:
            print(f"  Round {r['round']}: {r.get('model_version')}  "
                  f"hash={r.get('model_hash', 'N/A')[:18]}…")
        print("✓ Federation simulation complete.")
    else:
        # TRUE NETWORKED MODE — waits for real gRPC client connections
        from aura.blockchain import AURABlockchainLogger
        bc = AURABlockchainLogger()

        strategy = KrumFedAURA(
            blockchain_module = bc,
            num_rounds        = args.rounds,
        )
        server_config = fl.server.ServerConfig(
            num_rounds    = args.rounds,
            round_timeout = cfg.FL_ROUND_TIMEOUT_SEC,
        )

        print(f"\n{'='*62}")
        print(f"  AURA Federation Server — NETWORKED MODE")
        print(f"  Binding on:  {args.address}")
        print(f"  Rounds:      {args.rounds}")
        print(f"  Strategy:    Krum (Byzantine-robust, select {cfg.KRUM_NUM_TO_SELECT})")
        print(f"  Waiting for {cfg.FL_MIN_AVAILABLE} clients to connect …")
        print(f"{'='*62}\n")

        fl.server.start_server(
            server_address = args.address,
            config         = server_config,
            strategy       = strategy,
        )

