import copy
import torch
from torch import nn
import numpy as np
from GQLib.Optimizers.Neural_Network.utils_nn.lppls_nn_wrapper import LPPLSNNWrapper
from GQLib.filterings import LPPLSConfidence

conditions = LPPLSConfidence.BOUNDED_PARAMS

class BaseTrainer:
    """Base class for training neural networks on LPPLS data with early stopping."""
    def __init__(self,
                 t: np.ndarray,
                 x: np.ndarray,
                 net: nn.Module,
                 epochs: int = 3000,
                 lr: float = 1e-2,
                 device: str = "cpu",
                 val_ratio: float = 0.2,
                 patience: int = 10,
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
        self.epochs = epochs
        self.silent = silent
        self.val_ratio = val_ratio
        self.patience = patience

        # Train/validation split (contiguous)
        N = len(self.t)
        val_size = int(N * self.val_ratio)
        self.train_idx = slice(0, N - val_size)
        self.val_idx = slice(N - val_size, N)

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
        """Train the network with early stopping and return the results."""
        best_val_loss = float('inf')
        best_state = None
        wait = 0

        for epoch in range(1, self.epochs + 1):
            if not self.silent:
                print(f"\rEpoch {epoch}/{self.epochs}", end="")

            # ---- Train step ----
            self.opt.zero_grad()
            raw = self.net(self.t.unsqueeze(1))
            tc, m, w = self._map_params(raw)
            delta = tc - self.t
            mask = torch.isfinite(delta) & (delta > 0)
            # Train mask
            mask_train = mask[self.train_idx]
            if mask_train.sum() < 4:
                train_loss = torch.tensor(1e3, device=self.t.device)
            else:
                t_train = self.t[self.train_idx][mask_train]
                x_train = self.x[self.train_idx][mask_train]
                lppls_hat_train, _ = LPPLSNNWrapper._lppls_linear_params(
                    tc, m, w, t_train, x_train
                )
                train_loss = torch.mean((x_train - lppls_hat_train) ** 2)
            train_loss.backward()
            self.opt.step()

            # ---- Validation step ----
            with torch.no_grad():
                raw_val = self.net(self.t.unsqueeze(1))
                tc_v, m_v, w_v = self._map_params(raw_val)
                delta_v = tc_v - self.t
                mask_v = torch.isfinite(delta_v) & (delta_v > 0)
                mask_val = mask_v[self.val_idx]
                if mask_val.sum() < 4:
                    val_loss = torch.tensor(1e3, device=self.t.device)
                else:
                    t_val = self.t[self.val_idx][mask_val]
                    x_val = self.x[self.val_idx][mask_val]
                    lppls_hat_val, _ = LPPLSNNWrapper._lppls_linear_params(
                        tc_v, m_v, w_v, t_val, x_val
                    )
                    val_loss = torch.mean((x_val - lppls_hat_val) ** 2)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    if not self.silent:
                        print(f"\nEarly stopping at epoch {epoch}")
                    break

        # Reload best model
        if best_state is not None:
            self.net.load_state_dict(best_state)

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
            return tc_real, m_n, w_n, (A, B, C1, C2), float(best_val_loss)
        return tc_real, m_n, w_n, float(best_val_loss)
