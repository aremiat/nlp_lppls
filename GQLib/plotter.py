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

    def plot_subintervals(self, subintervals, sample, data):
        index_to_date = dict(zip(data[:, 0].astype(int), pd.to_datetime(data[:, 1])))

        sample_dates = np.array([index_to_date[int(idx)] for idx in sample[:, 0]])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])

        ax1.plot(sample_dates, sample[:, 1], 'b-', label='Price')
        ax1.set_title('Price Data with Subintervals')
        ax1.set_ylabel('Price')
        ax1.grid(True)

        sorted_subintervals = sorted(enumerate(subintervals), 
                                    key=lambda x: x[1][1] - x[1][0], 
                                    reverse=True)

        for i, (orig_idx, (start, end, _)) in enumerate(sorted_subintervals):
            start_date = index_to_date[int(start)]
            end_date = index_to_date[int(end)]
            duration = (end_date - start_date).days or 1  # fallback if same day
            ax2.barh(y=i, width=duration, left=start_date, height=1, alpha=0.5,
                    label=f'Interval {orig_idx+1}')

        ax2.set_title('Subintervals (Sorted by Length)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Subinterval Number')
        ax2.grid(True)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.show()

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