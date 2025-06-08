import torch
from torch import nn
from typing import Tuple

class LPPLSNNWrapper:
    """
    A flexible wrapper for LPPLS parameter estimation using a customizable neural network.
    """

    def __init__(self, net: nn.Module):
        """
        Initialize the wrapper with a neural network.

        Parameters:
            net (nn.Module): The neural network architecture to use.
        """
        if net is None:
            raise ValueError("A neural network instance must be provided.")
        self.net = net

    @staticmethod
    def _lppls_linear_params(tc: torch.Tensor, m: torch.Tensor, w: torch.Tensor,
                             t: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analytic least-squares for (A, B, C₁, C₂) given nonlinear params and series.
        """
        delta = tc - t
        mask = torch.isfinite(delta) & (delta > 0)
        if mask.sum() < 4:
            raise ValueError("Not enough valid points to solve for A, B, C1, C2.")

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

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Parameters:
            t (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Raw output from the neural network.
        """
        return self.net(t)