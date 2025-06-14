import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.io as pio
import os
from GQLib.Plotters.plotter import Plotter
from .Optimizers import Optimizer
from .subintervals import ClassicSubIntervals
from GQLib.LombAnalysis import LombAnalysis
from GQLib.Models import LPPL, LPPLS
from .enums import InputType
import logging
from GQLib.logging import with_spinner
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


logger = logging.getLogger(__name__)

with open("params/debug_framework.json", "r") as file:
    debug_params = json.load(file)

DEBUG_STATUS_GRAPH_TC = debug_params["DEBUG_STATUS_GRAPH_TC"]
DEBUG_STATUS_GRAPH_LOMB = debug_params["DEBUG_STATUS_GRAPH_LOMB"]

class Framework:
    """
    Framework for processing and analyzing financial time series using LPPL and Lomb-Scargle techniques.

    This framework includes:
    - Data loading and subinterval generation.
    - Optimization of LPPL parameters using a custom optimizer.
    - Lomb-Scargle periodogram analysis for detecting significant frequencies.
    - Visualization of results, including LPPL predictions and significant critical times.
    """

    def __init__(self, frequency: str = "daily", input_type : InputType = InputType.WTI, subinterval_method = ClassicSubIntervals) -> None:
        """
        Initialize the Framework with a specified frequency for analysis.

        Parameters
        ----------
        frequency : str, optional
            The frequency of the time series data. Must be one of {"daily", "weekly"}.
            Default is "daily".

        input_type : InputType, optional
            The  input type of the data selected

        Raises
        ------
        ValueError
            If an invalid frequency is provided.
        """
        # Frequency validation and data loading
        if frequency not in ["daily", "weekly", "monthly"]:
            raise ValueError("The frequency must be one of 'daily', 'weekly', 'monthly'.")
        
        self.frequency = frequency
        self.input_type = input_type
        self.subinterval_method = subinterval_method
        self.data = self.load_data()

        self.global_times = self.data[:, 0].astype(float)
        self.global_dates = self.data[:, 1]
        self.global_prices = self.data[:, 2].astype(float)

    @with_spinner("Loading data in progress ...")
    def load_data(self) -> np.ndarray:
        """
        Load financial time series data from a CSV file.

        The CSV file is expected to have two columns:
        - "Date": Date of observation in the format "%m/%d/%Y".
        - "Price": Observed price.

        The function adds a numeric time index and returns a NumPy array.

        Returns
        -------
        np.ndarray
            A 2D array of shape (N, 3), where:
            - Column 0: Numeric time index (float).
            - Column 1: Dates as np.datetime64[D].
            - Column 2: Prices as float.
        """
        match self.input_type:
            case InputType.USO:
                data = pd.read_csv(f'data/USO_{self.frequency}.csv', sep=";")
                data['Price'] = data['Price'].apply(lambda x:x/8) # Stock split 1:8 en 2020
                data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y").values.astype("datetime64[D]")

            case InputType.WTI:
                data = pd.read_csv(f'data/WTI_Spot_Price_{self.frequency}.csv', skiprows=4)
                data.columns = ["Date", "Price"]
                data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y").values.astype("datetime64[D]")

            case InputType.SP500:
                data = pd.read_csv(f'data/sp500_Price_daily.csv', sep=";")
                data.columns = ["Date", "Price"]
                data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y").values.astype("datetime64[D]")
            
            case InputType.BTC : 
                data = pd.read_csv(f'data/BTC_{self.frequency}.csv', sep=",")
                data.columns = ["Date", "Price"]
                data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d").values.astype("datetime64[D]")

            case InputType.SSE:
                data = pd.read_csv(f'data/SSE_Price_{self.frequency}.csv', sep=";")
                data.columns = ["Date", "Price"]
                data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y").values.astype("datetime64[D]")

            case InputType.EURUSD : 
                data = pd.read_csv(f'data/EURUSD_{self.frequency}.csv', sep=";")
                data.columns = ["Date", "Price"]
                data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d").values.astype("datetime64[D]")
            
        # Date conversion and sorting
        data = data.sort_values(by="Date")
        # Add numeric time index
        t = np.linspace(0, len(data) - 1, len(data))
        data = np.insert(data.to_numpy(), 0, t, axis=1)
        return data

    def process(self, time_start: str, time_end: str, optimizer: Optimizer) -> dict:
        """
        Optimize LPPL parameters over multiple subintervals of the selected sample.

        Parameters
        ----------
        time_start : str
            Start date of the main sample in "%d/%m/%Y" format.
        time_end : str
            End date of the main sample in "%d/%m/%Y" format.
        optimizer : Optimizer
            Optimizer instance for parameter fitting
        Returns
        -------
        dict
            Optimization results for each subinterval.
        """
        # Configure the params of the optimizer based on the frequency
        optimizer.configure_params_from_frequency(self.frequency, optimizer.__class__.__name__)
        # Select data sample
        sample = self.select_sample(self.data, time_start, time_end)

        # Generate subintervals
        subintervals = self.subinterval_method(sample).get_subintervals()

        # Store optimization results
        results = []

        # Optimize parameters for each subinterval
        with logging_redirect_tqdm():
            for sub_start, sub_end, sub_data in tqdm(
                subintervals,
                desc=f"Processing subintervals for {optimizer.__class__.__name__}",
                unit="subinterval",
            ):
                start = datetime.now()

                bestObjV, bestParams = optimizer.fit(sub_start, sub_end, sub_data)
                results.append({
                    "sub_start": sub_start,
                    "sub_end": sub_end,
                    "bestObjV": bestObjV,
                    "bestParams": bestParams.tolist(),
                    "time": (datetime.now() - start).total_seconds()
                })
        return results

    @with_spinner("Lomb-Scargle analysis in progress ...")
    def analyze(self,
                results: dict = None,
                result_json_name: str = None,
                lppl_model: 'LPPL | LPPLS' = LPPL,
                significativity_tc: float = 0.3,
                use_package: bool = False,
                remove_mpf: bool = True,
                mpf_threshold: float = 1e-3,
                show: bool = False) -> dict:
        """
        Analyze results using Lomb-Scargle periodogram and identify significant critical times.

        Parameters
        ----------
        results : dict
            Optimization results to analyze.
        result_json_name : dict, optional
            Path to a JSON file containing results. If None, uses `self.results`.
        lppl_model : 'LPPL | LPPLS'
            Log Periodic Power Law Model utilized to computer the Lomb Periodogram
        significativity_tc : float
            Significance Threshold for Frequency Closeness. Default is 0.3
        use_package : bool
            Whether to use the astropy package to compute the Lomb Periodogram Power
        remove_mpf : bool, optional
            Whether to remove the "most probable frequency" from the results. Default is True.
        mpf_threshold : float, optional
            Threshold for filtering frequencies close to the most probable frequency. Default is 1e-3.
        show : bool, optional
            Whether to display visualizations of the Lomb spectrum and LPPL fits. Default is False.
        Returns
        -------
        dict
            An updated list of results with significance flags.
        """
        if result_json_name is None and results is None:
            raise ValueError("Results must be provided.")

        if result_json_name is not None:
            with open(result_json_name, "r") as f:
                results = json.load(f)

        best_results = []

        # Visualizations if requested
        if show:
            num_intervals = len(results)
            num_cols = 3
            num_rows = (num_intervals + num_cols - 1) // num_cols
            fig, axes = plt.subplots(num_intervals, num_cols, figsize=(12, 6 * num_rows))

        for idx, res in enumerate(results):
            mask = (self.global_times >= res["sub_start"]) & (self.global_times <= res["sub_end"])
            t_sub = self.global_times[mask]
            y_sub = self.global_prices[mask]

            # Lomb-Scargle analysis
            lomb = LombAnalysis(lppl_model(t_sub, y_sub, res["bestParams"]))
            lomb.compute_lomb_periodogram(use_package=use_package)
            lomb.filter_results(remove_mpf=remove_mpf, mpf_threshold=mpf_threshold)
            is_significant = lomb.check_significance(significativity_tc=significativity_tc)

            if show:
                ax_residuals = axes[idx, 0]
                lomb.show_residuals(ax=ax_residuals)
                ax_residuals.set_title(f'Subinterval {idx + 1} Residuals')

                ax_spectrum = axes[idx, 1]
                lomb.show_spectrum(ax=ax_spectrum, use_filtered=True, show_threshold=True, highlight_freq=True)
                ax_spectrum.set_title(f'Subinterval {idx + 1} Spectrum (Significant: {is_significant})')

                ax_lppl = axes[idx, 2]
                self.show_lppl(lomb.lppl, ax=ax_lppl)
                ax_lppl.set_title(f'Subinterval {idx + 1} LPPL')

            # Add of the results
            best_results.append({
                "sub_start": res["sub_start"],
                "sub_end": res["sub_end"],
                "bestObjV": res["bestObjV"],
                "bestParams": res["bestParams"],
                "is_significant": is_significant,
                "power_value": max(lomb.power)
            })

        if show:
            plt.tight_layout()
            plt.show()

        return best_results

    @staticmethod
    def select_sample(data : np.asarray, time_start: str, time_end: str) -> np.ndarray:
        """
        Select a sample from the global time series based on a user-defined date range.

        Parameters
        ----------
        data : np.ndarray
            The global dataset as a NumPy array with columns: time index, date, and price.
        time_start : str
            The start date for the selection in the format "%d/%m/%Y".
        time_end : str
            The end date for the selection in the format "%d/%m/%Y".
        Returns
        -------
        np.ndarray
            A 2D array of shape (M, 2), where:
            - Column 0: Numeric time indices (float).
            - Column 1: Prices (float).
        """
        # Convert start and end dates to datetime64
        start_dt = np.datetime64(pd.to_datetime(time_start, format="%d/%m/%Y"))
        end_dt = np.datetime64(pd.to_datetime(time_end, format="%d/%m/%Y"))

        # Filter rows within the specified date range
        mask = (data[:, 1] >= start_dt) & (data[:, 1] <= end_dt)
        sample = data[mask]

        return sample[:, [0, 2]].astype(float)

    def plotter(self, plot_func_name: str, **kwargs):
        """
        Generic method to invoke any Plotter function by passing its specific arguments as keywords.

        Args:
            plot_func_name (str): Name of the Plotter method to call (e.g., "visualize_compare_results").
            **kwargs: Keyword arguments for the chosen Plotter function.
        """
        # Instancier Plotter avec les données dont il a besoin
        plotter = Plotter(
            global_dates=self.global_dates,
            global_prices=self.global_prices,
            frequency=self.frequency,
            input_type=self.input_type,
            save_image_func=self.save_image,
            real_tc=kwargs.get("real_tc", None),
        )

        if not hasattr(plotter, plot_func_name):
            raise ValueError(f"Plotter n’a pas de méthode '{plot_func_name}'.")
        method = getattr(plotter, plot_func_name)

        # On appelle la méthode de Plotter avec les kwargs fournis par l’utilisateur
        return method(**kwargs)

    @staticmethod
    def save_results(results: dict, file_name: str) -> None:
        """
        Save results to a JSON file.

        Parameters
        ----------
        results : dict
            Results to be saved.
        file_name : str
            Path to the output JSON file.
        """
        directory_path = os.path.dirname(file_name)

        if not os.path.exists(directory_path):
            logging.info(f"{directory_path} path was created !")
            os.makedirs(directory_path)


        with open(file_name, "w") as f:
            json.dump(results, f, indent=4)
    
    @staticmethod
    def save_image(fig , filename : str):
        """
        Save image to a png file.

        Parameters
        ----------
        fig : Figure
            Figure to be saved.
        filename : str
            Path to the output png file.
        """
        directory_path = os.path.dirname(filename)

        if not os.path.exists(directory_path):
            logging.info(f"{directory_path} path was created !")
            os.makedirs(directory_path)

        pio.write_image(fig, filename, scale=5, width=1000, height=800)

