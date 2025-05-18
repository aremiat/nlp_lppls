import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from GQLib.Models import LPPL, LPPLS
from typing import Optional, Union, List, Dict, Tuple


class Plotter:

    def __init__(self):
        pass

    def plot_subintervals(self):
        pass

    def plot_fitted_model(self):
        pass

    def plot_fitting_model(self):
        pass

    def plot_violin(self):
        pass

    def plot_kernel_density(self):
        pass

    def plot_lppl_fit(self, lppl: Union[LPPL, LPPLS], dates: np.array, prices: np.array):
        """
        Plot the fited LPPL model with observed prices.

        Parameters
            lppl (LPPL or LPPLS): The fitted LPPL model.

        """  
        length_extended = min(round(lppl.tc) + 1000, len(prices))
        extended_t = np.arange(lppl.t[0], length_extended)
        extended_y = prices[int(extended_t[0]):int(extended_t[-1] + 1)]
        extended_dates = dates[int(extended_t[0]):int(extended_t[-1] + 1)]
        end_date = dates[int(lppl.t[-1])]
        lppl.t = extended_t

        plt.plot(extended_dates, extended_y, label="Observed Prices", color="black", lw=1.5)
        plt.plot(extended_dates, lppl.predict(), label="LPPL Fit", color="red", linestyle="--", lw=2)

        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title("Observed Prices vs. LPPL Model Fit")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()