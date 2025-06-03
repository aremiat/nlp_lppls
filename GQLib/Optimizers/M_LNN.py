# m_lnn_updated.py – Mono‑LPPLS‑NN (M‑LNN) integrated with the existing Framework using AbstractNNOptimizer
# ---------------------------------------------------------------------------
# Layers:
#   1. Core neural network regressing the nonlinear LPPLS parameters (t_c, m, ω).
#   2. MLNNOptimizer subclass of AbstractNNOptimizer, drop‑in for Framework.process().
#   3. Convenience __main__ for standalone/demo usage.
# ---------------------------------------------------------------------------

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

# Import the new abstract optimizer interface
from .abstract_optimizer import AbstractNNOptimizer
from GQLib.Models import LPPLS

# ---------------------------------------------------------------------------
# 1. Core network + helper that solves the *linear* LPPLS parameters on the fly
# ---------------------------------------------------------------------------

torch.manual_seed(0)

def _lppls_linear_params(tc: torch.Tensor, m: torch.Tensor, w: torch.Tensor,
                         t: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytic least‑squares for (A, B, C₁, C₂) given nonlinear params and series."""
    delta = tc - t
    mask  = torch.isfinite(delta) & (delta > 0)
    if mask.sum() < 4:
        raise ValueError("Not enough valid points to solve for A,B,C1,C2.")

    t_sel, x_sel = t[mask], x[mask]
    f = delta[mask] ** m
    l = torch.log(delta[mask])
    g = f * torch.cos(w * l)
    h = f * torch.sin(w * l)

    M = torch.stack([torch.ones_like(t_sel), f, g, h], dim=1).double()
    y = x_sel.unsqueeze(1).double()

    theta, *_ = torch.linalg.lstsq(M, y, driver="gelsd")
    theta = theta.squeeze()
    A, B, C1, C2 = theta
    lppls_hat = (M @ theta.unsqueeze(1)).squeeze()
    return lppls_hat, torch.stack([A, B, C1, C2])


class _MLNNNet(nn.Module):
    """Two‑layer ReLU network → 3 outputs (t_c, m, ω)."""

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


class MLNNTrainer:
    """Train a Mono‑LPPLS‑NN on a *single* (t,x) series."""

    def __init__(self, t: np.ndarray, x: np.ndarray,
                 epochs: int = 3000, lr: float = 1e-2, device: str = "cpu",
                 silent: bool = False):
        # normalize time & price to [0,1]
        self.t0, self.t_max = t[0], t[-1]
        t_norm = (t - self.t0) / (self.t_max - self.t0)
        self.x_min, self.x_max = x.min(), x.max()
        x_scaled = (x - self.x_min) / (self.x_max - self.x_min)

        self.t = torch.tensor(t_norm, dtype=torch.float32, device=device)
        self.x = torch.tensor(x_scaled, dtype=torch.float32, device=device)

        self.net = _MLNNNet().to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.epochs, self.silent = epochs, silent

        # Hard bounds (normalized units)
        self.tc_min, self.tc_max = 1.0, 1.4
        self.m_min,  self.m_max  = 0.1, 1.0
        self.w_min,  self.w_max  = 6.0, 13.0

    def _map_params(self, raw: torch.Tensor) -> torch.Tensor:
        '''Squash raw outputs into allowed ranges via sigmoid.'''
        tc_r, m_r, w_r = raw
        tc = self.tc_min + (self.tc_max - self.tc_min) * torch.sigmoid(tc_r)
        m  = self.m_min  + (self.m_max  - self.m_min ) * torch.sigmoid(m_r)
        w  = self.w_min  + (self.w_max  - self.w_min ) * torch.sigmoid(w_r)
        return torch.stack([tc, m, w])

    def train(self, return_full: bool = False):
        t_col = self.t.unsqueeze(1)
        for epoch in range(1, self.epochs + 1):
            if not self.silent:
                print(f"\rEpoch {epoch}/{self.epochs}", end="")
            self.opt.zero_grad()
            tc, m, w = self._map_params(self.net(t_col))

            delta = tc - self.t
            mask = torch.isfinite(delta) & (delta > 0)
            if mask.sum() < 4:
                loss = torch.tensor(1e3, device=self.t.device)
            else:
                lppls_hat, _ = _lppls_linear_params(tc, m, w, self.t[mask], self.x[mask])
                loss = torch.mean((self.x[mask] - lppls_hat) ** 2)

            loss.backward()
            self.opt.step()

        tc_n, m_n, w_n = self._map_params(self.net(t_col)).detach().cpu().numpy()
        tc_real = self.t0 + tc_n * (self.t_max - self.t0)
        A, B, C1, C2 = _lppls_linear_params(
            torch.tensor(tc_n), torch.tensor(m_n), torch.tensor(w_n), self.t, self.x
        )[1].numpy()

        if not self.silent:
            print("\nEstimated parameters:")
            print(f"  t_c   ≈ {tc_real:.2f}")
            print(f"  m     ≈ {m_n:.4f}")
            print(f"  omega ≈ {w_n:.4f}\n")

        if return_full:
            return tc_real, m_n, w_n, (A, B, C1, C2), float(loss.detach())
        return tc_real, m_n, w_n, float(loss.detach())

# ---------------------------------------------------------------------------
# 2. MLNNOptimizer subclassing AbstractNNOptimizer
# ---------------------------------------------------------------------------

class MLNNOptimizer(AbstractNNOptimizer):
    '''Drop‑in replacement for GA/PSO/SA optimizers inside Framework.'''

    def __init__(self, lppl_model, epochs: int = 3000, lr: float = 1e-2, device: str = "cpu"):
        self.epochs, self.lr, self.device = epochs, lr, device
        self.name = "MLNN"
        self.lppl_model = lppl_model

    def configure_params_from_frequency(self, *args, **kwargs):
        '''No extra parameter bounds needed for MLNN.'''
        pass

    def fit(self, sub_start: float, sub_end: float, sub_data: np.ndarray):
        '''Fit on subinterval and return (loss, bestParams).'''
        t, y = sub_data[:,0], sub_data[:,1]
        trainer = MLNNTrainer(t, y, epochs=self.epochs, lr=self.lr, device=self.device, silent=True)
        tc, m, w, loss = trainer.train(return_full=False)
        return loss, np.array([tc, m, w])