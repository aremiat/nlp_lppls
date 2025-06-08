from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Import abstract optimizer
from GQLib.Optimizers.abstract_optimizer import Optimizer
from GQLib.Models import LPPLS
from GQLib.filterings import LPPLSConfidence
from .base_trainer import BaseTrainer

torch.manual_seed(0)
conditions = LPPLSConfidence.BOUNDED_PARAMS


class NLCNNLPPLSNet(nn.Module):
    """Nonlinear CNN1D for LPPLS: applies exponent-weighted conv before ReLU."""
    def __init__(self,
                 channels: int = 16,
                 kernel_size: int = 3,
                 num_layers: int = 2,
                 eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.convs = nn.ModuleList()
        self.exp_weights = nn.ParameterList()
        in_ch = 1
        for _ in range(num_layers):
            conv = nn.Conv1d(in_ch, channels, kernel_size, padding=kernel_size//2)
            # exponent weight matrix same shape as conv weight, shared across batch
            exp_w = nn.Parameter(torch.ones_like(conv.weight))
            self.convs.append(conv)
            self.exp_weights.append(exp_w)
            in_ch = channels
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, 3)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (N,1) -> (batch=1, channels=1, length=N)
        x = t.unsqueeze(0).permute(0,2,1)
        # apply each nonlinear conv layer
        for conv, exp_w in zip(self.convs, self.exp_weights):
            # calculez un nouveau noyau pondéré
            weighted_kernel = conv.weight * torch.exp(exp_w)
            # puis utilisez directement F.conv1d
            x = F.conv1d(x, weighted_kernel, bias=conv.bias, padding=conv.padding[0])
            x = F.relu(x)
        # pool and fc
        feat = self.pool(x).squeeze(-1)  # shape (1, channels)
        raw = self.fc(feat)              # shape (1,3)
        return raw.squeeze(0)            # (3,)


class NLCNNTrainer(BaseTrainer):
    pass


class NLCNN(Optimizer):
    """Optimizer using NLCNN for LPPLS."""
    def __init__(self,
                 lppl_model: 'LPPL | LPPLS' = LPPLS,
                 net: nn.Module | None = None,
                 epochs: int = 3000,
                 lr: float = 1e-2,
                 device: str = "cpu"):
        self.lppl_model, self.device = lppl_model, device
        self.net = net if net is not None else NLCNNLPPLSNet()
        self.epochs, self.lr = epochs, lr
        self.name = "NLCNN-LPPLS"

    def fit(self,
            sub_start: float,
            sub_end: float,
            sub_data: np.ndarray) -> Tuple[float, np.ndarray]:
        t, y = sub_data[:,0], sub_data[:,1]
        trainer = NLCNNTrainer(t,y, net=self.net,
                               epochs=self.epochs, lr=self.lr,
                               device=self.device, silent=True)
        tc, m, w, loss = trainer.train(return_full=False)
        return loss, np.array([tc, m, w])

