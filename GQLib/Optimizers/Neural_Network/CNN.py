from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
from torch import nn

from GQLib.Optimizers.abstract_optimizer import Optimizer
from GQLib.Models import LPPLS
from GQLib.Optimizers.Neural_Network.base_trainer import BaseTrainer


torch.manual_seed(0)

# CNN

class CNNLPPLSNet(nn.Module):
    """Réseau CNN1D qui prédit les paramètres non-linéaires (t_c, m, ω)."""
    def __init__(self, channels: int = 16, kernel_size: int = 3, num_layers: int = 2):
        super().__init__()
        layers = []
        in_ch = 1
        for i in range(num_layers):
            layers.append(nn.Conv1d(in_ch, channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            in_ch = channels
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, 3)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (N,1) -> (batch=1, channels=1, length=N)
        x = t.unsqueeze(0).permute(0,2,1)
        features = self.conv(x)
        pooled = self.pool(features).squeeze(-1)
        raw = self.fc(pooled)
        return raw.squeeze(0)  # Renvoie un vecteur de taille 3


class CNNTrainer(BaseTrainer):
    """Trainer for CNN-based LPPLS models."""
    pass  # Inherits all functionality from BaseTrainer

# ---------------------------------------------------------------------------
# 2. CNNOptimizer
# ---------------------------------------------------------------------------


class CNN(Optimizer):
    """Optimizer based on CNN for LPPL/LPPLS."""
    def __init__(self,
                 lppl_model: 'LPPL | LPPLS' = LPPLS,
                 net: nn.Module | None = None,
                 epochs: int = 3000,
                 lr: float = 1e-2,
                 device: str = "cpu",
                 silent: bool = True):
        self.epochs, self.lr, self.device = epochs, lr, device
        self.name = "CNN-LPPLS"
        self.lppl_model = lppl_model
        self.net = net if net is not None else CNNLPPLSNet()
        self.silent = silent

    def fit(self,
            sub_start: float,
            sub_end: float,
            sub_data: np.ndarray) -> Tuple[float, np.ndarray]:
        t, y = sub_data[:, 0], sub_data[:, 1]
        trainer = CNNTrainer(
            t, y,
            net=self.net,
            epochs=self.epochs,
            lr=self.lr,
            device=self.device,
            silent=self.silent
        )
        tc, m, w, loss = trainer.train(return_full=False)
        return loss, np.array([tc, m, w])