import numpy as np
import pandas as pd
import json
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from datetime import datetime
import plotly.io as pio
import plotly.graph_objects as go
import os
from GQLib.plotter import Plotter
from .Optimizers import MPGA, PSO, SGA, SA, Optimizer
from .subintervals import ClassicSubIntervals, DidierSubIntervals, SubIntervalMethod
from GQLib.LombAnalysis import LombAnalysis
from GQLib.Models import LPPL, LPPLS
from .enums import InputType
from typing import Optional, Union, List, Dict, Tuple
import logging
from GQLib.logging import with_spinner
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde
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

    def _base(self, start_date: str, end_date: str, real_tc: str = None, title: str = None) -> tuple:
        """
        Trace le prix + t1, t2, real_tc et retourne fig, ax
        avec zorder élevés par défaut.
        """
        # conversion des dates
        start_training = pd.to_datetime(start_training, format="%d/%m/%Y")
        end_training   = pd.to_datetime(end_date,      format="%d/%m/%Y")

        # fenêtre
        window_start = start_training - timedelta(days=90)
        window_end   = end_training   + timedelta(days=730)

        # extraction
        mask   = [(window_start <= d <= window_end) for d in self.global_dates]
        dates  = [d for d, m in zip(self.global_dates, mask) if m]
        prices = [p for p, m in zip(self.global_prices, mask) if m]

        # création figure/axe
        fig, ax = plt.subplots(figsize=(18, 8))

        # rendre la figure et l'axe transparents
        fig.patch.set_alpha(0)     # fond de la figure
        ax.patch.set_alpha(0)      # fond de l'axe

        # prix
        ax.plot(dates, prices,
                color="black", linewidth=1.2,
                label=f"{self.input_type.value} {self.frequency} price",
                zorder=20)

        # bornes t1/t2
        ax.axvline(x=start_training, color="black",
                linestyle="-.", linewidth=1, label="t1",
                zorder=21)
        ax.axvline(x=end_training,   color="black",
                linestyle="-.", linewidth=1, label="t2",
                zorder=21)
        # fill between the all space between t1 and t2
        ax.axvspan(start_training, end_training, facecolor="gray", alpha=0.15)

        # real_tc
        if real_tc is not None:
            if isinstance(real_tc, str):
                real_tc = pd.to_datetime(real_tc, format="%d/%m/%Y")
            elif isinstance(real_tc, int):
                real_tc = self.global_dates[real_tc]
            ax.axvline(x=real_tc, color="red",
                    linewidth=2, label="Real critical time",
                    zorder=22)

        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title("Log-Price & LPPLS fits")
            
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{self.input_type.value} {self.frequency} price")
        leg = ax.legend(loc="upper left", title="Algorithmes / bornes")
        leg.get_frame().set_alpha(0.9)
        # Puis on remonte la légende
        leg.set_zorder(25)

        plt.tight_layout()
        return fig, ax

    def _add_kernels(self, fig, ax, dict_results):
        """
        Add kernels to the plot.

        Parameters
        ----------
        fig : Figure
            The figure object.
        ax : Axes
            The axes object.
        dict_results : dict
            Dictionary containing the results for each kernel.
        """
        colors = [
            "#ffa15a",  # Orange clair
            "#ab63fa",  # Violet clair
            "#00cc96",  # Vert clair
            "#ef553b",  # Rouge clair
            "#636efa",  # Bleu clair
            "#19d3f3",  # Cyan
            "#ff6692",  # Rose clair
            "#b6e880",  # Vert lime
            "#ff97ff",  # Magenta clair
        ]

        # Crée un second axe pour tracer les densités
        ax2 = ax.twinx()
        ax2.set_ylabel("Densité de tc", fontsize=12)

        for idx, (opt, values) in enumerate(dict_results.items()):
            print(values)
            # 1) récupère la distribution brute
            distrib_all = values["tc_distrib"]
            # 2) convertit chaque "index" flottant en date
            rounded = [int(round(i)) for i in distrib_all]
            kernel_dates = [self.global_dates[i] for i in rounded]

            # 3) passe les dates en nombres matplotlib (float)
            numeric_dates = mdates.date2num(kernel_dates)

            # 4) estime la densité de noyau
            kde = gaussian_kde(numeric_dates)
            x_eval = np.linspace(numeric_dates.min(), numeric_dates.max(), 200)
            y_eval = kde(x_eval)

            # 5) trace la densité
            ax2.plot(
                x_eval,
                y_eval,
                color=colors[idx % len(colors)],
                linewidth=1.5,
                label=opt
            )

        # légende et alignement des limites en x
        ax2.legend(title="Algorithmes", loc="upper right", fontsize=10)
        ax2.set_xlim(ax.get_xlim())

        # formate automatiquement les dates pour les deux axes
        fig.autofmt_xdate()

    def _add_half_violins(self, fig, ax, dict_results,
                        width_scale: float = 0.5,
                        spacing: float = 0.5,
                        specific: str = "tc_distrib",
                        hatch_pattern="/",
                        color: str = "white",
                        text: bool = False):

        # 1) créer l'axe secondaire
        ax_v = ax.twinx()
        # 2) le placer SOUS l'axe principal
        ax_v.set_zorder(0)        # twin axis lowest
        ax.set_zorder(1)          # main axis above
        # 3) rendre son fond transparent
        ax_v.patch.set_alpha(0)
        ax_v.set_yticks([])
        ax_v.set_ylabel("")

        # 4) préparer la grille de dates (inchangé)
        all_num = []
        for vals in dict_results.values():
            raw = vals[specific]
            idxs = [int(round(i)) for i in raw]
            dates = [self.global_dates[i] for i in idxs]
            all_num.append(mdates.date2num(dates))

        mn, mx = min(arr.min() for arr in all_num), max(arr.max() for arr in all_num)
        date_grid = np.linspace(mn, mx, 200)

        max_y = -np.inf

        # 5) tracer les demi-violons en zorder bas
        for idx, (opt, vals) in enumerate(dict_results.items()):
            raw = vals[specific]
            idxs = [int(round(i)) for i in raw]
            dates = [self.global_dates[i] for i in idxs]
            num = mdates.date2num(dates)

            kde = gaussian_kde(num)
            dens = kde(date_grid)
            dens = dens / dens.max() * width_scale
            y0 = idx * spacing
            hatch = hatch_pattern
            # ligne de base
            ax_v.hlines(y=y0, xmin=mn, xmax=mx,
                        colors="black", linewidth=0.5)
            # demi-violon blanc + hachures
            poly = ax_v.fill_between(
                date_grid, y0, y0 + dens,
                facecolor=color, edgecolor="black",
                linewidth=0.8, alpha=0.90,
            )
            poly.set_hatch(hatch)
            # contour supérieur
            ax_v.plot(date_grid, y0 + dens,
                    color="black", linewidth=1.0)
            
            # label
            if text:
                ax_v.text(mx + (mx - mn)*0.01, y0 + dens.max()*0.5,
                        opt.replace("_", "\n"), va="center", ha="left")

            max_y = max(max_y, y0 + dens.max())

        # 6) ajuster l'axe Y et formater X
        ax_v.set_ylim(-spacing*0.5, max_y + spacing*0.5)
        ax_v.xaxis_date()
        fig.autofmt_xdate()

        # légende (facultative, ici on utilise plutôt les labels texte)
        ax.legend(loc="upper left", title="Algorithmes")

    def _add_lppl_fit(self, fig, ax, dict_results: dict, nb_calib: int = 3, window_extension: int = 1000):
        """
        Ajoute les courbes de fit LPPL/LPPLS au graphique de base.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            La figure sur laquelle on travaille.
        ax : matplotlib.axes.Axes
            L'axe principal (celui des prix) pour y superposer les fits.
        lppl_models : list of LPPL or LPPLS
            Liste d'instances de modèles LPPL(S) déjà calibrés.
        """
        calib_set = dict_results["Set 1"]["NELDER_MEAD"]["raw_run_result"]
        indices = random.sample(range(len(calib_set)), nb_calib)
        calib_set_selected = [calib_set[i] for i in indices]

        for i in range(nb_calib):
            info = calib_set_selected[i]
            mask   = [(info["sub_start"] <= t <= info["sub_end"]) for t in self.global_times]
            prices = [p for p, m in zip(self.global_prices, mask) if m]
            
            lppl = LPPLS(params=info["bestParams"], 
                         t= np.linspace(info["sub_start"], info["sub_end"], len(prices)),
                         y=np.array(prices))
            
            mask   = [(info["sub_start"] <= t <= info["sub_end"] + window_extension) for t in self.global_times]
            dates = [p for p, m in zip(self.global_dates, mask) if m]
            lppl.t = np.array([p for p, m in zip(self.global_times, mask) if m])

            ax.plot(
                dates,
                lppl.predict(),
                linestyle="--",
                linewidth=2,
                label=f"{lppl.__name__} (tc={self.global_dates[int(lppl.tc)].strftime('%d-%m-%Y')})",
            )

        ax.legend(title="LPPL Fits", loc="upper left", fontsize=10)
        fig.autofmt_xdate()
