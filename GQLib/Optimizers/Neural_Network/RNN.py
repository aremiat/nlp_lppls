from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
from torch import nn

# Import de l'interface optimiseur abstrait
from GQLib.Optimizers.abstract_optimizer import Optimizer
from GQLib.Models import LPPLS
from GQLib.Optimizers.Neural_Network.base_trainer import BaseTrainer


torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 1. Réseau RNN + helper résolvant les paramètres *linéaires* LPPLS à la volée
# ---------------------------------------------------------------------------


class RNNLPPLSNet(nn.Module):
    """Réseau RNN (LSTM) qui prédit les paramètres non-linéaires (t_c, m, ω)."""
    def __init__(self, hidden_size: int = 16, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.LSTM(input_size=1,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (N,1) => (batch=1, seq_len=N, features=1)
        batch = t.unsqueeze(0)
        outputs, (h_n, c_n) = self.rnn(batch)
        # Utiliser la dernière sortie de la séquence
        last = outputs[:, -1, :]
        raw = self.fc(last)
        return raw.squeeze(0)  # Renvoie un vecteur de taille 3


class RNNTrainer(BaseTrainer):
    """Trainer for RNN-based LPPLS models."""
    pass
# ---------------------------------------------------------------------------
# 2. RNN hérité d'AbstractNNOptimizer
# ---------------------------------------------------------------------------


class RNN(Optimizer):
    """Optimiseur basé sur RNN pour LPPL/LPPLS."""
    def __init__(self,
                 lppl_model: 'LPPL | LPPLS' = LPPLS,
                 net: nn.Module | None = None,
                 epochs: int = 3000,
                 lr: float = 1e-2,
                 device: str = "cpu"):
        self.epochs, self.lr, self.device = epochs, lr, device
        self.name = "RNN-LPPLS"
        self.lppl_model = lppl_model
        # Si aucun réseau fourni, instancier un RNN par défaut
        self.net = net if net is not None else RNNLPPLSNet()

    def fit(self,
            sub_start: float,
            sub_end: float,
            sub_data: np.ndarray) -> Tuple[float, np.ndarray]:
        t, y = sub_data[:, 0], sub_data[:, 1]
        trainer = RNNTrainer(
            t, y,
            net=self.net,
            epochs=self.epochs,
            lr=self.lr,
            device=self.device,
            silent=True
        )
        tc, m, w, loss = trainer.train(return_full=False)
        return loss, np.array([tc, m, w])