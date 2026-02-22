"""
aura/fl_client.py — Flower Federated Learning Client
=====================================================

Each "organisation" (Bank, Hospital, ISP) runs one instance of this client.
The client owns a LOCAL copy of the AURAModelBundle and trains it on its
own private data.  Only the mathematical weight deltas (gradients) are ever
sent to the server — raw data NEVER leaves the local network.

Federation Lifecycle (per round)
---------------------------------
1. Server → Client: broadcasts current global model weights
2. Client: loads weights into local model
3. Client: trains for LOCAL_EPOCHS on local data partition
4. Client: sends updated weights back to server
5. Server: applies Krum aggregation to drop potential poisoned updates

Privacy Guarantee:
  Differential Privacy (DP) is the production extension.
  For the hackathon demo, we demonstrate the architectural boundary —
  no raw data (IP logs, user records) leave the client boundary.
"""

import io
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from flwr.common import (
    Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes,
    GetParametersIns, GetParametersRes, Status, Code,
    ndarrays_to_parameters, parameters_to_ndarrays,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from aura.models import AURAModelBundle

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Model ↔ NumPy Parameter Conversion
# ─────────────────────────────────────────────────────────────────────────────

def model_to_ndarrays(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays (Flower format)."""
    return [p.detach().cpu().numpy() for p in model.parameters()]


def ndarrays_to_model(model: nn.Module, arrays: List[np.ndarray]) -> None:
    """Load a list of NumPy arrays into model parameters (in-place)."""
    with torch.no_grad():
        for p, arr in zip(model.parameters(), arrays):
            p.copy_(torch.tensor(arr))


# ─────────────────────────────────────────────────────────────────────────────
# AURA Flower Client
# ─────────────────────────────────────────────────────────────────────────────

class AURAFlowerClient(fl.client.Client):
    """
    Flower client that encapsulates a local AURAModelBundle and its training
    data partition (representing one organisation's private network).

    Parameters
    ----------
    client_id      : Unique identifier for this client (e.g., "hospital_1")
    train_data     : Tensor[N_local, F] — local normalised flow features
    val_data       : Tensor[M_local, F] — local validation split
    local_epochs   : Number of local SGD epochs per federation round
    device         : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        client_id:    str,
        train_data:   torch.Tensor,
        val_data:     torch.Tensor,
        local_epochs: int   = 3,
        device:       str   = "cpu",
    ):
        self.client_id    = client_id
        self.train_data   = train_data.to(device)
        self.val_data     = val_data.to(device)
        self.local_epochs = local_epochs
        self.device       = device

        # Local model — each org starts with a fresh copy; federation aligns them
        self.model    = AURAModelBundle().to(device)
        self.optimizer = torch.optim.Adam(
            self.model.autoencoder.parameters(),
            lr=cfg.AE_LEARNING_RATE,
        )
        logger.info(f"[{client_id}] Flower client initialised  |  "
                    f"train={len(train_data)}  val={len(val_data)}  epochs={local_epochs}")

    # ------------------------------------------------------------------
    # Flower Protocol Methods
    # ------------------------------------------------------------------

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return current local model weights to the server."""
        arrays = model_to_ndarrays(self.model)
        return GetParametersRes(
            status     = Status(code=Code.OK, message="OK"),
            parameters = ndarrays_to_parameters(arrays),
        )

    def fit(self, ins: FitIns) -> FitRes:
        """
        Receive global weights, train locally, return updated weights.

        Step 1: Overwrite local model with received global weights
        Step 2: Run LOCAL_EPOCHS of unsupervised autoencoder training
        Step 3: Return updated parameters + training metadata
        """
        logger.info(f"[{self.client_id}] Round started — loading global weights …")

        # Step 1: Load global model
        global_arrays = parameters_to_ndarrays(ins.parameters)
        ndarrays_to_model(self.model, global_arrays)

        # Step 2: Local training on private data
        num_examples, train_loss = self._local_train()

        # Step 3: Return updated weights
        updated_arrays = model_to_ndarrays(self.model)
        logger.info(f"[{self.client_id}] Round complete  |  "
                    f"loss={train_loss:.4f}  examples={num_examples}")

        return FitRes(
            status     = Status(code=Code.OK, message="OK"),
            parameters = ndarrays_to_parameters(updated_arrays),
            num_examples = num_examples,
            metrics    = {"train_loss": float(train_loss), "client_id": 0},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the received global weights on local validation data."""
        arrays = parameters_to_ndarrays(ins.parameters)
        ndarrays_to_model(self.model, arrays)

        self.model.autoencoder.eval()
        with torch.no_grad():
            x_hat, _ = self.model.autoencoder(self.val_data)
            loss = nn.functional.mse_loss(x_hat, self.val_data)

        logger.info(f"[{self.client_id}] Eval loss: {loss.item():.4f}")
        return EvaluateRes(
            status       = Status(code=Code.OK, message="OK"),
            loss         = float(loss),
            num_examples = len(self.val_data),
            metrics      = {"val_loss": float(loss)},
        )

    # ------------------------------------------------------------------
    # Local Training
    # ------------------------------------------------------------------

    def _local_train(self) -> Tuple[int, float]:
        """
        Run unsupervised autoencoder training on local data.

        We train in batch mode — the autoencoder learns to reconstruct
        the local network's 'normal' flow distribution.  If this client's
        network gets attacked, the reconstruction error will spike, and
        the updated weights (incorporating the new attack-learned boundary)
        will be shared with the federation.

        Returns:  (num_training_examples, final_batch_loss)
        """
        ae = self.model.autoencoder
        ae.train()

        dataset = torch.utils.data.TensorDataset(self.train_data)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.AE_BATCH_SIZE, shuffle=True
        )

        last_loss = 0.0
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                self.optimizer.zero_grad()
                x_hat, z = ae(batch)
                loss      = ae.reconstruction_loss(batch, x_hat, z)
                loss.backward()
                # Gradient clipping: prevents exploding gradients with
                # adversarially crafted data (a known FL poisoning vector)
                torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            last_loss = epoch_loss / max(len(loader), 1)
            logger.debug(f"  [{self.client_id}] epoch={epoch+1}  loss={last_loss:.4f}")

        return len(self.train_data), last_loss


# ─────────────────────────────────────────────────────────────────────────────
# Client Factory (creates mock clients for the demo)
# ─────────────────────────────────────────────────────────────────────────────

def create_mock_clients(
    n_clients:    int   = 3,
    n_samples:    int   = 500,
    feature_dim:  int   = cfg.FEATURE_DIM,
    attack_client: int  = None,     # None = randomly poison one; -1 = all honest
    org_ids:      list  = None,   # Override org IDs e.g. ["hospital","university"]
) -> List["AURAFlowerClient"]:
    """
    Factory function for hackathon demo.

    Creates N mock clients with synthetic Gaussian flow data.
    One client (attack_client index) has data poisoned to simulate a real
    network under attack — this is what gives Krum a genuine outlier to detect.

    Parameters
    ----------
    attack_client : Index of the client to poison.
                    None  → randomly select one org to inject attack traffic (default)
                    -1    → all clients train honestly (no Byzantine signal)
                    0-N   → explicitly poison that client index
    org_ids       : Optional list of org keys ["hospital","bank","university"]
                    overriding the default 3-client set.  Length must match n_clients.
    """
    import random as _random

    _default_orgs = ["hospital", "bank", "university"]
    if org_ids is None:
        org_ids = _default_orgs[:n_clients]

    # Randomly inject attack data into one org so Krum has a real signal
    if attack_client is None:
        attack_client = _random.randint(0, len(org_ids) - 1)
        logger.info(f"[MOCK] Attack data injected into index {attack_client} "
                    f"({org_ids[attack_client]}) — Krum should detect this outlier")
    elif attack_client == -1:
        attack_client = None   # all clients honest — Krum drop is arbitrary

    clients = []
    for i, org_key in enumerate(org_ids):
        client_id = f"org_{org_key}_1"

        # Base data: Normal traffic
        train_data = torch.rand(n_samples, feature_dim) * 0.3 + 0.35

        if i == attack_client:
            n_attack = n_samples // 5
            attack_rows = torch.rand(n_attack, feature_dim)
            attack_rows[:, [2, 3, 15]] = torch.rand(n_attack, 3) * 0.3 + 0.7
            attack_rows[:, [4, 5, 63]] = torch.rand(n_attack, 3) * 0.2 + 0.8
            train_data[:n_attack] = attack_rows
            logger.info(f"[{client_id}] Attack simulation injected (Byzantine).")

        val_data = torch.rand(n_samples // 5, feature_dim) * 0.3 + 0.35
        clients.append(AURAFlowerClient(client_id, train_data, val_data))

    return clients, attack_client   # return who was selected as Byzantine


# ─────────────────────────────────────────────────────────────────────────────
# Networked Client Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def start_client(
    client_id:      str,
    server_address: str = cfg.FL_SERVER_ADDRESS,
    n_samples:      int = 500,
    is_byzantine:   bool = False,
) -> None:
    """
    Start a Flower gRPC client that connects to the FL server over the network.

    This is the REAL networked FL entry point.  Each organisation's gateway
    switch runs this function in its own process — the client dials the
    aggregation server via gRPC, trains locally, and sends only weight deltas
    (raw data NEVER leaves the network).

    Parameters
    ----------
    client_id      : human-readable org identifier (e.g. "org_hospital_1")
    server_address : host:port of the FL aggregation server
    n_samples      : number of local flow records to train on
    is_byzantine   : if True, injects attack-pattern data (adversarial client)
    """
    import flwr as fl

    feature_dim = cfg.FEATURE_DIM

    # ── Simulate each org's local network traffic distribution ──────────────
    train_data = torch.rand(n_samples, feature_dim) * 0.3 + 0.35
    val_data   = torch.rand(n_samples // 5, feature_dim) * 0.3 + 0.35

    if is_byzantine:
        # Adversarial client: poisoned data with extreme feature values
        n_attack = n_samples // 5
        attack_rows = torch.rand(n_attack, feature_dim)
        attack_rows[:, [2, 3, 15]] = torch.rand(n_attack, 3) * 0.3 + 0.7
        attack_rows[:, [4, 5, 63]] = torch.rand(n_attack, 3) * 0.2 + 0.8
        train_data[:n_attack] = attack_rows
        logger.info(f"[{client_id}] Byzantine mode — poisoned data injected.")

    client = AURAFlowerClient(client_id, train_data, val_data)

    print(f"\n[{client_id}] Connecting to FL server at {server_address} …")
    print(f"[{client_id}] Network: {'ADVERSARIAL (Byzantine)' if is_byzantine else 'Normal'}")
    print(f"[{client_id}] Local dataset: {n_samples} flow records  |  features: {feature_dim}")

    fl.client.start_client(
        server_address = server_address,
        client         = client.to_client(),
    )
    print(f"[{client_id}] Federation complete. Local model updated.")


# CLI entry point — called by run_federation_networked.py per-process
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AURA FL Client (networked mode)")
    parser.add_argument("--client-id",  required=True,  help="e.g. org_hospital_1")
    parser.add_argument("--server",     default=cfg.FL_SERVER_ADDRESS, help="host:port")
    parser.add_argument("--samples",    type=int, default=500)
    parser.add_argument("--byzantine",  action="store_true", help="Adversarial client")
    parser.add_argument("--network-sim",default="",   help="Simulated LAN CIDR (display only)")
    args = parser.parse_args()

    if args.network_sim:
        print(f"[{args.client_id}] Simulated network: {args.network_sim}")

    start_client(
        client_id      = args.client_id,
        server_address = args.server,
        n_samples      = args.samples,
        is_byzantine   = args.byzantine,
    )
