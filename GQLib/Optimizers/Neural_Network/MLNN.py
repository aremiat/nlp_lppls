from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
from torch import nn

# Import the new abstract optimizer interface
from GQLib.Optimizers.abstract_optimizer import Optimizer
from GQLib.Models import LPPLS, LPPL
from GQLib.Optimizers.Neural_Network.base_trainer import BaseTrainer

# ---------------------------------------------------------------------------
# 1. Core network + helper that solves the *linear* LPPLS parameters on the fly
# ---------------------------------------------------------------------------

torch.manual_seed(0)


class _MLNNNet(nn.Module):
    def __init__(self, n_hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(1, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, 3)
        self.relu = nn.ReLU()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(t))
        h = self.relu(self.fc2(h))
        raw = self.out(h)
        return raw.mean(dim=0)


# Trainer for Mono‑LPPLS‑NN (M‑LNN) on a single (t,x) series
class MLNNTrainer(BaseTrainer):
    """Train a Mono‑LPPLS‑NN on a *single* (t,x) series."""
    pass


# ---------------------------------------------------------------------------
# 2. MLNN subclassing AbstractNNOptimizer
# ---------------------------------------------------------------------------

class MLNN(Optimizer):
    """Neural network optimizer for LPPL/LPPLS using a custom architecture."""

    def __init__(self, lppl_model: 'LPPL | LPPLS' = LPPL, net: nn.Module = None,
                 epochs: int = 3000, lr: float = 1e-2, device: str = "cpu"):
        """
        Initialize the MLNN.

        Parameters:
            lppl_model (LPPL | LPPLS): The LPPL/LPPLS model to use.
            net (nn.Module): Custom neural network architecture.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
            device (str): Device to use ('cpu' or 'cuda').
        """
        self.epochs, self.lr, self.device = epochs, lr, device
        self.name = "MLNN"
        self.lppl_model = lppl_model
        self.net = net if net is not None else _MLNNNet()

    def fit(self, sub_start: float, sub_end: float, sub_data: np.ndarray):
        """
        Fit on subinterval and return (loss, bestParams).

        Parameters:
            sub_start (float): Start of the subinterval.
            sub_end (float): End of the subinterval.
            sub_data (np.ndarray): Subinterval data (time and price).

        Returns:
            Tuple[float, np.ndarray]: Loss and best parameters (t_c, m, ω).
        """
        t, y = sub_data[:, 0], sub_data[:, 1]
        trainer = MLNNTrainer(t, y, net=self.net, epochs=self.epochs,
                              lr=self.lr, device=self.device, silent=True)
        tc, m, w, loss = trainer.train(return_full=False)
        return loss, np.array([tc, m, w])