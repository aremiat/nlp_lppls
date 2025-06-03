from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

class AbstractFilter(ABC):

    @abstractmethod
    def filter(self, model_params: list) -> bool:
        """
        Filter the time series estimation based on the model parameters.

        Parameters:
            model_params (list): list of model parameters [t_c, omega, alpha]

        Returns:
            bool: True if the time series is filtered, False otherwise
        """
        pass

class enculefilter(AbstractFilter):

    def filter():
        return True
    
class LPPLSConfidence(AbstractFilter):

    SEARCH_SPACE = {
        "alpha": [0, 2.0],
        "omega": [1, 50],
        "t_c": [],
    }

    CONDITIONS_EARLY_BUBBLE = {
        "alpha": [0.01, 1.2],
        "omega": [2, 25],
        "t_c": [-0.2, 0.2],
        "nb_oscillations": [2.5, np.inf],
        "damping": [0.0, np.inf],
    }

    CONDITIONS_EARLY_BUBBLE = {
        "alpha": [0.01, 1.2],
        "omega": [2, 25],
        "t_c": [-0.05, 0.1],
        "nb_oscillations": [2.5, np.inf],
        "damping": [0.0, np.inf],
    }

    CONDITIONS_BUBBLE_END = {
        "alpha": [0.01, 1.2],
        "omega": [2, 25],
        "t_c": [-0.05, 0.1],
        "nb_oscillations": [2.5, np.inf],
        "damping": [0.0, np.inf],
    }

    def __init__(self, model: 'LPPLS', len_window: int, **kwargs: Any) -> None:
        self.model = model
        self.len_window = len_window
        self.kwargs = kwargs

    def get_search_space(self) -> Dict[str, List[float]]:
        bounds = self.SEARCH_SPACE.copy()
        bounds["t_c"] = [self.len_window * b for b in bounds["t_c"]]
        if self.model is LPPL:
            bounds["phi"] = [0, 2 * np.pi]
        return self.SEARCH_SPACE
    
    def compute_damping(self, alpha: float, B: float, omega: float, C) -> float:
        """
        Compute the damping factor based on the LPPLS parameters.
        """
        return (alpha * omega) / (2 * np.pi) * np.exp(-omega * t_c)

    def is_valid(self, linear_params: list, non_linear_params: list, dt: float) -> bool:
        """
        Validate the parameters based on LPPLS-specific criteria, including dynamic bounds for t_c.

        Parameters:
            linear_params (list): List of linear parameters.
            non_linear_params (list): List of non-linear parameters.
            dt (float): Time step or length of the series.

        Returns:
            bool: True if the parameters meet the criteria, False otherwise.
        """
        alpha, omega, t_c = non_linear_params
        t_c_bounds = [-0.02 * dt, 0.02 * dt]  # Dynamic bounds for t_c

        return (
            self.CONDITIONS_BUBBLE_END["alpha"][0] <= alpha <= self.CONDITIONS_BUBBLE_END["alpha"][1] and
            self.CONDITIONS_BUBBLE_END["omega"][0] <= omega <= self.CONDITIONS_BUBBLE_END["omega"][1] and
            t_c_bounds[0] <= t_c <= t_c_bounds[1]
        )