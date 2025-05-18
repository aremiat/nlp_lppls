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

    def plot_lppl(self, lppl: Union[LPPL, LPPLS], dates: np.array, prices: np.array):
        """
        Plot the fited LPPL model with observed prices.

        Parameters
            lppl (LPPL or LPPLS): The fitted LPPL model.

        """
        sns.set(style="whitegrid")

        length_extended = min(round(lppl.tc) + 1000, len(prices))

        extended_t = np.arange(lppl.t[0], length_extended)
        extended_y = prices[int(extended_t[0]):int(extended_t[-1] + 1)]
        extended_dates = prices[int(extended_t[0]):int(extended_t[-1] + 1)]
        end_date = prices[int(lppl.t[-1])]

        lppl.t = extended_t
        predicted = lppl.predict(True)

        fig, ax = plt.subplots(figsize=(12, 6))

        palette = sns.color_palette("tab10")

        ax.plot(extended_dates, extended_y, label='Observed', color=palette[0], linewidth=2)
        ax.plot(extended_dates, predicted, label='Predicted', color=palette[1], linewidth=2, linestyle='--')
        ax.axvline(x=end_date, color='red', linestyle=':', linewidth=1.5, label='End of Subinterval')

        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Prévision du modèle LPPL', fontsize=14, fontweight='bold')

        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', rotation=45)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

        plt.tight_layout()
        plt.show()
