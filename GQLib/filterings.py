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


class LPPLSConfidence(AbstractFilter):

    BOUNDED_PARAMS = {
        "alpha": [0.1, 1.0],
        "omega": [6, 13],
        "t_c": [1.01, 1.8],
    }

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


    def is_valid(self, non_linear_params: list, dt: float) -> bool:
        """
        Validate the parameters based on LPPLS-specific criteria, including dynamic bounds for t_c.

        Parameters:
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