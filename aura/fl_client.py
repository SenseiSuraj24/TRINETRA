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
    attack_client: int  = 1,     # This client injects "attack-pattern" data
) -> List[AURAFlowerClient]:
    """
    Factory function for hackathon demo.

    Creates N mock clients with synthetic Gaussian flow data.
    One client (attack_client index) has their data slightly perturbed
    to simulate a network under attack — their local model will learn
    this attack pattern and share the fingerprint via federation.
    """
    clients = []
    for i in range(n_clients):
        client_id = f"org_{['hospital', 'bank', 'university'][i % 3]}_{i + 1}"

        # Base data: Normal traffic — centred at 0.5 in normalised feature space
        train_data = torch.rand(n_samples, feature_dim) * 0.3 + 0.35

        if i == attack_client:
            # Inject 20% attack-like samples: bimodal distribution with
            # extreme values in specific feature dimensions (e.g., byte ratios)
            n_attack = n_samples // 5
            attack_rows = torch.rand(n_attack, feature_dim)
            # DDoS-like: extreme packet rates in dims [2, 3, 15]
            attack_rows[:, [2, 3, 15]] = torch.rand(n_attack, 3) * 0.3 + 0.7
            # Exfil-like: abnormal byte transfer in dims [4, 5, 63]
            attack_rows[:, [4, 5, 63]] = torch.rand(n_attack, 3) * 0.2 + 0.8
            train_data[:n_attack] = attack_rows
            logger.info(f"[{client_id}] Attack simulation injected into training data.")

        val_data = torch.rand(n_samples // 5, feature_dim) * 0.3 + 0.35
        clients.append(AURAFlowerClient(client_id, train_data, val_data))

    return clients


# CLI test handle — see fl_server.py for the full federation run
if __name__ == "__main__":
    print("=== AURA FL Client — Mock Test ===")
    clients = create_mock_clients(n_clients=3)
    for c in clients:
        print(f"  Created: {c.client_id}  |  train={len(c.train_data)}")
    print("✓ FL clients created.")
