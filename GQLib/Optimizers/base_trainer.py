import torch
from torch import nn
import numpy as np
from .lppls_nn_wrapper import LPPLSNNWrapper
from GQLib.filterings import LPPLSConfidence

conditions = LPPLSConfidence.BOUNDED_PARAMS


class BaseTrainer:
    """Base class for training neural networks on LPPLS data."""
    def __init__(self,
                 t: np.ndarray,
                 x: np.ndarray,
                 net: nn.Module,
                 epochs: int = 3000,
                 lr: float = 1e-2,
                 device: str = "cpu",
                 silent: bool = False):

        # Normalize time and price
        self.t0, self.t_max = t[0], t[-1]
        t_norm = (t - self.t0) / (self.t_max - self.t0)
        self.x_min, self.x_max = x.min(), x.max()
        x_scaled = (x - self.x_min) / (self.x_max - self.x_min)

        self.t = torch.tensor(t_norm, dtype=torch.float32, device=device)
        self.x = torch.tensor(x_scaled, dtype=torch.float32, device=device)

        self.net = net
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.epochs, self.silent = epochs, silent

        # Parameter bounds
        self.tc_min, self.tc_max = conditions["t_c"]
        self.m_min, self.m_max = conditions["alpha"]
        self.w_min, self.w_max = conditions["omega"]

    def _map_params(self, raw: torch.Tensor) -> torch.Tensor:
        """Map raw network outputs to parameter bounds using sigmoid."""
        tc_r, m_r, w_r = raw
        tc = self.tc_min + (self.tc_max - self.tc_min) * torch.sigmoid(tc_r)
        m  = self.m_min  + (self.m_max  - self.m_min ) * torch.sigmoid(m_r)
        w  = self.w_min  + (self.w_max  - self.w_min ) * torch.sigmoid(w_r)
        return torch.stack([tc, m, w])

    def train(self, return_full: bool = False):
        """Train the network and return the results."""
        for epoch in range(1, self.epochs + 1):
            if not self.silent:
                print(f"\rEpoch {epoch}/{self.epochs}", end="")
            self.opt.zero_grad()
            raw = self.net(self.t.unsqueeze(1))
            tc, m, w = self._map_params(raw)

            delta = tc - self.t
            mask = torch.isfinite(delta) & (delta > 0)
            if mask.sum() < 4:
                loss = torch.tensor(1e3, device=self.t.device)
            else:
                lppls_hat, _ = LPPLSNNWrapper._lppls_linear_params(
                    tc, m, w, self.t[mask], self.x[mask]
                )
                loss = torch.mean((self.x[mask] - lppls_hat) ** 2)

            loss.backward()
            self.opt.step()

        # Extract final parameters
        raw = self.net(self.t.unsqueeze(1)).detach()
        tc_n, m_n, w_n = self._map_params(raw).cpu().numpy()
        tc_real = self.t0 + tc_n * (self.t_max - self.t0)
        A, B, C1, C2 = LPPLSNNWrapper._lppls_linear_params(
            torch.tensor(tc_n), torch.tensor(m_n), torch.tensor(w_n), self.t, self.x
        )[1].numpy()

        if not self.silent:
            print("\nEstimated parameters:")
            print(f"  t_c   ≈ {tc_real:.2f}")
            print(f"  m     ≈ {m_n:.4f}")
            print(f"  omega ≈ {w_n:.4f}\n")

        if return_full:
            return tc_real, m_n, w_n, (A, B, C1, C2), float(loss.item())
        return tc_real, m_n, w_n, float(loss.item())