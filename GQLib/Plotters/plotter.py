import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from GQLib.Models import LPPL, LPPLS
from typing import Optional, Union, List, Dict, Tuple
import logging
import plotly.graph_objects as go
from datetime import datetime, timedelta
from GQLib.logging import with_spinner

import plotly.io as pio
pio.renderers.default = 'browser'



class Plotter:

    def __init__(
        self,
        global_dates: np.ndarray,
        global_prices: np.ndarray,
        frequency: str,
        input_type,
        save_image_func,
        real_tc: Optional[str] = None,
    ):
        self.global_dates = global_dates
        self.global_prices = global_prices
        self.frequency = frequency
        self.input_type = input_type
        self.save_image = save_image_func
        self.real_tc = real_tc

    @with_spinner("Creation of visualization in progress ...")
    def visualize_compare_results(self, multiple_results: dict[str, dict],
                                  name: str = "",
                                  data_name: str = "",
                                  real_tc: str = None,
                                  optimiseurs_models: list = None,
                                  start_date: str = None,
                                  end_date: str = None,
                                  nb_tc: int = 20,
                                  save_plot: bool = False):
        """
        Visualize and compare multiple optimizers results on the same period
        Args:
            multiple_results (dict[str, dict]): dictionnary of results to display
            name (str, optional): Name of the graph Defaults to "".
            data_name (str, optional): name of the data. Defaults to "".
            real_tc (str, optional): The real tc to display.
            optimiseurs_models (list, optional): Optimizers Models .
            start_date (str, optional): start date of the computing interval.
            end_date (str, optional): end date of the computing interval. Defaults to None.
            nb_tc (int, optional): Number of tc necessary to calcul the exact tc. Defaults to 20.
            save_plot (bool, optional): Whether to save the plot. Defaults to False.
        """

        # Adapt multiple_result to old format :
        temp = {}
        for key, value in multiple_results.items():
            temp[key] = value["raw_filtered_result"]
        multiple_results = temp

        logging.debug("\n Visualize function input :")
        logging.debug(f"multiple_results : {multiple_results}")
        logging.debug(f"name : {name}")
        logging.debug(f"data_name : {data_name}")
        logging.debug(f"real_tc : {real_tc}")
        logging.debug(f"optimiseurs_models : {optimiseurs_models}")
        logging.debug(f"start_date : {start_date}")
        logging.debug(f"end_date : {end_date}")
        logging.debug(f"nb_tc : {nb_tc}")
        logging.debug(f"save_plot : {save_plot}\n")

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
        logging.debug("Starting visualize_compare_results with start_date=%s and end_date=%s", start_date, end_date)
        start = start_date
        end = end_date

        name_plot = ""
        if start_date is not None and end_date is not None:
            try:
                start_date = pd.to_datetime(start_date, format="%d/%m/%Y")
                end_date = pd.to_datetime(end_date, format="%d/%m/%Y") + timedelta(days=365)
                logging.debug("Parsed start_date: %s, end_date: %s", start_date, end_date)
            except Exception as e:
                logging.error("Error parsing start_date or end_date: %s", e)
                raise
        else:
            start_date = np.min(self.global_dates)
            end_date = np.max(self.global_dates)
            logging.warning("start_date or end_date is None. Using global date range: %s to %s", start_date, end_date)
        # Filtration
        filtered_indices = [i for i, date in enumerate(self.global_dates) if start_date <= date <= end_date]
        if not filtered_indices:
            logging.info("Aucune donnée disponible entre %s et %s.", start_date, end_date)
            return

        filtered_dates = [self.global_dates[i] for i in filtered_indices]
        filtered_prices = [self.global_prices[i] for i in filtered_indices]
        logging.debug("Filtered %d data points for visualization", len(filtered_dates))

        fig = go.Figure()
        # Plot de la série de prix
        fig.add_trace(go.Scatter(x=filtered_dates, y=filtered_prices, mode='lines', name=data_name,
                                 line=dict(color="black", width=1)))
        logging.debug("Base price series plotted")

        # Si la vraie date du tc est fournie, on la plot
        if self.real_tc is not None:
            try:
                target_date = pd.to_datetime(self.real_tc, format="%d/%m/%Y")
                logging.debug("Parsed real critical time: %s", target_date)
            except Exception as e:
                logging.error("Error parsing real_tc: %s", e)
            target_date = None
            if target_date:
                fig.add_trace(
                    go.Scatter(
                        x=[target_date, target_date],
                        y=[min(filtered_prices), max(filtered_prices)],
                        mode="lines",
                        line=dict(color="red", width=4),
                        name="Real critical time",
                        showlegend=True
                    )
                )
            logging.debug("Real critical time plotted at %s", target_date)

        # Je veux garder 1/5 du max de la time series en haut et en bas
        total_height = max(filtered_prices) - min(filtered_prices)
        base_y = total_height / 6
        remaining_height = total_height - 2 * base_y
        # On divise l'espace restant pour que chaque modèle ait la même hauteur
        rectangle_height = remaining_height / len(multiple_results.keys())
        logging.debug("Calculated rectangle_height: %s", rectangle_height)

        for i, (optimizer_name, results) in enumerate(multiple_results.items()):
            logging.debug("Processing optimizer: %s", optimizer_name)
            # Récupération du modèle LPPL correspondant
            lppl_model_name = optimiseurs_models[i] if optimiseurs_models and i < len(
                optimiseurs_models) else "Unknown Model"
            legend_label = f"{optimizer_name} ({lppl_model_name})"
            name_plot += f"{optimizer_name}({lppl_model_name})_"
            best_results = results
            significant_tc = []
            min_time = np.inf
            max_time = -np.inf

            for res in best_results:
                if res["sub_start"] < min_time:
                    min_time = res["sub_start"]
                if res["sub_end"] > max_time:
                    max_time = res["sub_end"]
                if res["is_significant"]:
                    significant_tc.append([res["bestParams"][0], res["power_value"]])
            logging.debug("Optimizer '%s': min_time=%s, max_time=%s, significant_tc=%s", optimizer_name, min_time,
                          max_time, significant_tc)

            try:
                if (nb_tc is not None):
                    significant_tc = sorted(significant_tc, key=lambda x: x[1], reverse=True)[
                                     :min(len(significant_tc), nb_tc)]
                    logging.debug("Trimmed significant_tc: %s", significant_tc)
                # Calcul de la date exacte du tc en pondérant nb_tc par leur power
                sum_max_power = sum(x[1] for x in significant_tc if x[1] is not None and not np.isnan(x[1]))
                weighted_sum_tc = sum(x[0] * x[1] for x in significant_tc if x[1] is not None and not np.isnan(x[1]))
                significant_tc = weighted_sum_tc / sum_max_power if sum_max_power != 0 else 0
                logging.debug("Computed weighted significant tc: %s", significant_tc)
            except Exception as e:
                logging.error("Error processing significant_tc for optimizer '%s': %s", optimizer_name, e)
                continue

            # On plot les start et end date une fois à la première itération
            if i == 0:
                if start_date <= self.global_dates[int(min_time)] <= end_date:
                    fig.add_trace(go.Scatter(x=[self.global_dates[int(min_time)], self.global_dates[int(min_time)]],
                                             y=[min(filtered_prices), max(filtered_prices)], mode="lines",
                                             line=dict(color="gray", dash="dash"), name="Start Date", showlegend=True))
                    logging.debug("Start Date plotted")
                if start_date <= self.global_dates[int(max_time)] <= end_date:
                    fig.add_trace(go.Scatter(x=[self.global_dates[int(max_time)], self.global_dates[int(max_time)]],
                                             y=[min(filtered_prices), max(filtered_prices)], mode="lines",
                                             line=dict(color="gray", dash="longdash"), name="End Date",
                                             showlegend=True))
                    logging.debug("End Date plotted")

            # Calcul des dates des tc
            if significant_tc and isinstance(significant_tc, float):
                logging.info("Model '%s'", optimizer_name)
                if len(self.global_dates) > significant_tc > 0:
                    logging.info("Significant TC : %s", self.global_dates[int(round(significant_tc))])
                    min_tc_date = self.global_dates[int(round(significant_tc))] - timedelta(days=15)
                    max_tc_date = self.global_dates[int(round(significant_tc))] + timedelta(days=15)
                elif significant_tc > len(self.global_dates):
                    extra_dates_needed = int(significant_tc) - len(self.global_dates) + 1
                    last_date = self.global_dates.max()
                    freq = "B" if self.frequency == "daily" else "W"
                    new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=extra_dates_needed,
                                              freq=freq)
                    min_tc_date = new_dates[extra_dates_needed - 1] - timedelta(days=15)
                    max_tc_date = new_dates[extra_dates_needed - 1] + timedelta(days=15)
                    logging.debug("New dates created for tc beyond range")
                else:
                    logging.info("No significant TC found, or out of range for optimizer '%s'", optimizer_name)
                    continue

                # Rectangle pour le modèle
                fig.add_trace(go.Scatter(
                    x=[min_tc_date, max_tc_date, max_tc_date, min_tc_date, min_tc_date],
                    y=[min(filtered_prices) + base_y + i * rectangle_height,
                       min(filtered_prices) + base_y + i * rectangle_height,
                       min(filtered_prices) + base_y + i * rectangle_height + rectangle_height,
                       min(filtered_prices) + base_y + i * rectangle_height + rectangle_height,
                       min(filtered_prices) + base_y + i * rectangle_height],
                    fill="toself", fillcolor=colors[i % len(colors)], opacity=0.5, showlegend=True,
                    mode="lines+markers", marker=dict(size=1),
                    line=dict(color="gray", width=1), name=legend_label))
                logging.debug("Rectangle plotted for optimizer '%s'", optimizer_name)

                # Ajout du nom du modèle au centre du rectangle
                center_x = min_tc_date + (max_tc_date - min_tc_date) / 2
                center_y = min(filtered_prices) + base_y + i * rectangle_height + rectangle_height / 2
                fig.add_trace(
                    go.Scatter(x=[center_x], y=[center_y], text=[optimizer_name], mode="text", showlegend=False))
                logging.debug("Label added at center for optimizer '%s'", optimizer_name)

                fig.update_layout(title=name,
                                  xaxis=dict(
                                      title='Date',
                                      showline=True,
                                      linecolor='black',
                                      linewidth=1,
                                      mirror=True
                                  ),
                                  yaxis=dict(
                                      title=f"{self.input_type.value} {self.frequency} price",
                                      showline=True,
                                      linecolor='black',
                                      linewidth=1,
                                      mirror=True
                                  ),
                                  showlegend=True,
                                  plot_bgcolor='white',
                                  paper_bgcolor='white')
        pio.renderers.default = 'browser'
        fig.show()
        logging.info("Figure displayed")
        if save_plot:
            try:
                start_date_obj = datetime.strptime(start, "%d/%m/%Y")
                end_date_obj = datetime.strptime(end, "%d/%m/%Y")
                filename = f"results_{self.input_type.value}/algo_comparison//{self.frequency}/{name_plot}{start_date_obj.strftime('%m-%Y')}_{end_date_obj.strftime('%m-%Y')}.png"
                self.save_image(fig, filename)
                logging.info("Figure saved at %s", filename)
            except Exception as e:
                logging.error("Error saving figure: %s", e)

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

        def visualize_tc(self,
                         best_results: dict,
                         name="",
                         data_name: str = "",
                         start_date: str = None,
                         end_date: str = None,
                         nb_tc: int = None,
                         real_tc: str = None) -> None:
            """
            Visualize significant critical times on the price series.
            Allows filtering and displaying results for a specific date range.

            Args:
                best_results (dict): Optimal results containing information about the turning points.
                name (str): Name of the graph.
                data_name (str) : Name of the data
                start_date (str): Start date (format: 'YYYY-MM-DD'). If None, uses the start of the data.
                end_date (str): End date (format: 'YYYY-MM-DD'). If None, uses the end of the data.
                nb_tc (int): Maximum number of turning points to display.
                real_tc (str): Actual value of the turning point.
            """

            logging.debug("\n Visualize function input :")
            logging.debug(f"best_results : {best_results}")
            logging.debug(f"name : {name}")
            logging.debug(f"data_name : {data_name}")
            logging.debug(f"start_date : {start_date}")
            logging.debug(f"end_date : {end_date}")
            logging.debug(f"nb_tc : {nb_tc}")
            logging.debug(f"real_tc : {real_tc}\n")

            significant_tc = []
            min_time = np.inf
            max_time = -np.inf

            logging.info("Visualisation des tc")

            if start_date is not None:
                start_date = pd.to_datetime(start_date, format="%d/%m/%Y")
            else:
                start_date = np.min(self.global_dates)

            if end_date is not None:
                end_date = pd.to_datetime(end_date, format="%d/%m/%Y")
            else:
                end_date = np.max(self.global_dates)

            filtered_indices = [
                i for i, date in enumerate(self.global_dates) if start_date <= date <= end_date
            ]
            if not filtered_indices:
                logging.info(f"Aucune donnée disponible entre {start_date} et {end_date}.")
                return

            filtered_dates = [self.global_dates[i] for i in filtered_indices]
            filtered_prices = [self.global_prices[i] for i in filtered_indices]
            fig = go.Figure()
            # Plot de la série de prix
            fig.add_trace(go.Scatter(x=filtered_dates, y=filtered_prices, mode='lines', name=data_name,
                                     line=dict(color="black", width=1)))

            # Si la vraie date du tc est fournie, on la plot
            if real_tc is not None:
                target_date = pd.to_datetime(real_tc, format="%d/%m/%Y")

                fig.add_trace(
                    go.Scatter(
                        x=[target_date, target_date],
                        y=[min(filtered_prices), max(filtered_prices)],
                        mode="lines",
                        line=dict(color="green", width=4),
                        name="Real critical time",
                        showlegend=True
                    )
                )

            for res in best_results:
                if res["sub_start"] < min_time:
                    min_time = res["sub_start"]
                if res["sub_end"] > max_time:
                    max_time = res["sub_end"]
                if res["is_significant"]:
                    significant_tc.append([res["bestParams"][0], res["power_value"]])

            # Add of computing start date and end date
            if start_date <= self.global_dates[int(min_time)] <= end_date:
                fig.add_trace(go.Scatter(x=[self.global_dates[int(min_time)], self.global_dates[int(min_time)]],
                                         y=[min(filtered_prices), max(filtered_prices)], mode="lines",
                                         line=dict(color="gray", dash="dash"), name="Start Date", showlegend=True))

            if start_date <= self.global_dates[int(max_time)] <= end_date:
                fig.add_trace(go.Scatter(x=[self.global_dates[int(max_time)], self.global_dates[int(max_time)]],
                                         y=[min(filtered_prices), max(filtered_prices)], mode="lines",
                                         line=dict(color="gray", dash="longdash"), name="End Date", showlegend=True))

            try:
                if (nb_tc != None):
                    # Select the number of tc
                    significant_tc = sorted(significant_tc, key=lambda x: x[1], reverse=True)[:nb_tc]
                    significant_tc = [element[0] for element in significant_tc]

                else:
                    significant_tc = [element[0] for element in significant_tc]
            except:
                pass

            index_plot = 0
            for tc in significant_tc:
                try:
                    date_tc = self.global_dates[int(round(tc))]
                    if start_date <= date_tc <= end_date:
                        fig.add_trace(
                            go.Scatter(
                                x=[date_tc, date_tc],
                                y=[min(filtered_prices), max(filtered_prices)],
                                mode="lines",
                                line=dict(color="red", dash="dot"),
                                name=f"Critical times" if index_plot == 0 else None,
                                showlegend=(index_plot == 0)
                            )
                        )
                        index_plot += 1
                except:
                    continue

            fig.update_layout(title=name,
                              xaxis=dict(
                                  title='Date',
                                  showline=True,
                                  linecolor='black',
                                  linewidth=1,
                                  mirror=True
                              ),
                              yaxis=dict(
                                  title=f"{self.input_type.value} {self.frequency} price",
                                  showline=True,
                                  linecolor='black',
                                  linewidth=1,
                                  mirror=True
                              ),
                              showlegend=True,
                              plot_bgcolor='white',
                              paper_bgcolor='white')
            fig.show()

            def show_lppl(self, lppl: 'LPPL | LPPLS', ax=None, show: bool = False) -> None:
                """
                Visualize the LPPL or LPPLS fit alongside observed data.

                Parameters
                ----------
                lppl : LPPL or LPPLS
                    An instance of the LPPL or LPPLS model with fitted parameters.
                ax : matplotlib.axes.Axes, optional
                    An axis to plot on. If None, creates a new figure.
                show : bool, optional
                    Whether to display the plot immediately. Default is False.
                """
                length_extended = (round(lppl.tc) + 1000) if self.frequency == "daily" else (round(lppl.tc) + 100)

                # Calculate the maximum available length
                max_length = len(self.global_prices)

                # Adjust length_extended so it does not exceed the available length
                length_extended = min(length_extended, max_length)

                extended_t = np.arange(lppl.t[0], length_extended)
                extended_y = self.global_prices[int(extended_t[0]):int(extended_t[-1] + 1)]
                extended_dates = self.global_dates[int(extended_t[0]):int(extended_t[-1] + 1)]
                end_date = self.global_dates[int(lppl.t[-1])]

                lppl.t = extended_t
                predicted = lppl.predict(True)

                if ax is None:
                    fig, ax = plt.subplots(figsize=(10, 6))

                ax.plot(extended_dates, extended_y, label='Observed')
                ax.plot(extended_dates, predicted, label='Predicted')
                ax.axvline(x=end_date, color='r', linestyle='--', label='End of Subinterval')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.set_title('LPPL Model Prediction')
                ax.legend()

                if show:
                    plt.show()
                #     plotter = Plotter()
                #     plotter.plot_lppl_fit(lppl, self.global_dates, self.global_prices)
    @staticmethod
    def plot_loss(losses: list[float], model_name = None) -> None:
            """
            Plot the loss function over epochs using Plotly.

            Args:
                losses (list[float]): List of loss values for each epoch.
                save_plot (bool, optional): Whether to save the plot. Defaults to False.
                filename (str, optional): File path to save the plot. Defaults to None.
            """
            epochs = list(range(1, len(losses) + 1))
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=epochs,
                y=losses,
                mode='lines+markers',
                name='Loss',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))

            fig.update_layout(
                title="Loss Function Over Epochs" + (f" - {model_name}" if model_name else ""),
                xaxis_title="Epochs",
                yaxis_title="Loss",
                template="plotly_white",
                showlegend=True
            )

            fig.show()