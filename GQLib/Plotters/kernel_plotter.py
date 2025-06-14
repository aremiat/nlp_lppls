# kernel_plotter.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from scipy.stats import gaussian_kde
import logging
from typing import Any, Dict, List, Tuple, Union, Optional

logger = logging.getLogger(__name__)


class KernelPlotter:
    """Visualise price, LPPLS windows, tc kernel densities & half-violins."""

    # ---------------- construction ----------------
    def __init__(self,
                 dates: List[pd.Timestamp],
                 prices: List[float],
                 input_type: str = "Asset",
                 frequency: str = "daily"):
        self.global_dates  = dates
        self.global_prices = prices
        self.input_type    = input_type   # e.g. "WTI"
        self.frequency     = frequency    # e.g. "daily"

    @classmethod
    def from_framework(cls, fw):
        # 1) dates
        dates = getattr(fw, "global_dates", None)
        if dates is None:
            dates = getattr(fw, "dates", None)
        if dates is None:
            raise AttributeError("Framework must expose .global_dates or .dates")

        # 2) prices
        prices = getattr(fw, "global_prices", None)
        if prices is None:
            prices = getattr(fw, "prices", None)
        if prices is None:
            raise AttributeError("Framework must expose .global_prices or .prices")

        # 3) input_type
        input_type = getattr(fw, "input_type", "Asset")
        if hasattr(input_type, "value"):  # Enum → .value
            input_type = input_type.value

        # 4) frequency
        frequency = getattr(fw, "frequency", "daily")

        return cls(dates, prices, input_type=input_type, frequency=frequency)

    # ---------------- figure de base ----------------
    def _base(self,
              start_date: str,
              end_date:   str,
              real_tc:    Union[str, int, None] = None,
              title:      Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        start_training = pd.to_datetime(start_date, format="%d/%m/%Y")
        end_training   = pd.to_datetime(end_date,   format="%d/%m/%Y")
        window_start   = start_training - timedelta(days=90)
        window_end     = end_training   + timedelta(days=730)

        mask   = [(window_start <= d <= window_end) for d in self.global_dates]
        dates  = [d for d, m in zip(self.global_dates, mask) if m]
        prices = [p for p, m in zip(self.global_prices, mask) if m]

        fig, ax = plt.subplots(figsize=(18, 8))
        fig.patch.set_alpha(0); ax.patch.set_alpha(0)

        ax.plot(dates, prices, color="black", lw=1.2,
                label=f"{self.input_type} {self.frequency} price", zorder=20)

        ax.axvline(start_training, color="black", ls="-.", lw=1, label="t1", zorder=21)
        ax.axvline(end_training,   color="black", ls="-.", lw=1, label="t2", zorder=21)
        ax.axvspan(start_training, end_training, facecolor="gray", alpha=0.15)

        if real_tc is not None:
            if isinstance(real_tc, str):
                real_tc = pd.to_datetime(real_tc, format="%d/%m/%Y")
            elif isinstance(real_tc, int):
                real_tc = self.global_dates[real_tc]
            ax.axvline(real_tc, color="red", lw=2, label="Real critical time", zorder=22)

        ax.set_title(title or "Log-Price & LPPLS fits")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{self.input_type} {self.frequency} price")
        leg = ax.legend(loc="upper left", title="Algorithmes / bornes")
        leg.get_frame().set_alpha(0.9); leg.set_zorder(25)
        plt.tight_layout()
        return fig, ax

    # ---------------- densités kernel ----------------
    def _add_kernels(self, fig: plt.Figure, ax: plt.Axes,
                     dict_results: Dict[str, Dict]):
        colors = ["#ffa15a", "#ab63fa", "#00cc96", "#ef553b", "#636efa",
                  "#19d3f3", "#ff6692", "#b6e880", "#ff97ff"]
        ax2 = ax.twinx(); ax2.set_ylabel("Densité de tc")
        if not dict_results:
            logger.warning("No results provided for kernel density estimation.")
            return

        for idx, (opt, values) in enumerate(dict_results.items()):
            distrib_all = values.get("tc_distrib", [])
            numeric = [mdates.date2num(self.global_dates[int(round(i))])
                       for i in distrib_all if int(round(i)) < len(self.global_dates)]
            if not numeric:
                continue
            kde   = gaussian_kde(numeric)
            grid  = np.linspace(min(numeric), max(numeric), 200)
            ax2.plot(grid, kde(grid),
                     color=colors[idx % len(colors)], lw=1.5, label=opt)

        ax2.legend(title="Algorithmes", loc="upper right", fontsize=10)
        ax2.set_xlim(ax.get_xlim()); fig.autofmt_xdate()

    # ---------------- demi-violons ----------------
    def _add_half_violins(self, fig: plt.Figure, ax: plt.Axes,
                          dict_results: Dict[str, Dict],
                          width_scale: float = 0.5, spacing: float = 0.5,
                          specific: str = "tc_distrib",
                          hatch_pattern: str = "/",
                          color: str = "white",
                          text: bool = False):
        ax_v = ax.twinx(); ax_v.set_zorder(0); ax.set_zorder(1)
        ax_v.patch.set_alpha(0); ax_v.set_yticks([]); ax_v.set_ylabel("")
        if not dict_results:
            logger.warning("No results provided for half-violins."); return

        all_num = []
        for v in dict_results.values():
            num = [mdates.date2num(self.global_dates[int(round(i))])
                   for i in v[specific] if int(round(i)) < len(self.global_dates)]
            all_num.append(np.array(num))
        mn, mx = min(arr.min() for arr in all_num), max(arr.max() for arr in all_num)
        date_grid = np.linspace(mn, mx, 200)

        max_y = -np.inf
        for idx, (opt, vals) in enumerate(dict_results.items()):
            num = [mdates.date2num(self.global_dates[int(round(i))])
                   for i in vals[specific] if int(round(i)) < len(self.global_dates)]
            kde  = gaussian_kde(num)
            dens = kde(date_grid); dens = dens / dens.max() * width_scale
            y0   = idx * spacing
            ax_v.hlines(y=y0, xmin=mn, xmax=mx, colors="black", lw=0.5)
            poly = ax_v.fill_between(date_grid, y0, y0 + dens,
                                     facecolor=color, edgecolor="black",
                                     lw=0.8, alpha=0.90)
            poly.set_hatch(hatch_pattern)
            ax_v.plot(date_grid, y0 + dens, color="black", lw=1.0)

            if text:
                ax_v.text(mx + (mx - mn)*0.01,
                          y0 + dens.max()*0.5,
                          opt.replace("_", "\n"),   # ← correction ici
                          va="center", ha="left")
            max_y = max(max_y, y0 + dens.max())

        ax_v.set_ylim(-spacing*0.5, max_y + spacing*0.5)
        ax_v.xaxis_date(); fig.autofmt_xdate()

        # ---------------- courbes LPPL(S) calibrées ----------------
    def _add_lppl_fit(self,
                          fig: plt.Figure,
                          ax: plt.Axes,
                          dict_results: Dict,
                          nb_calib: int = 3,
                          window_ext: int = 1_000):
            """
            Superpose quelques courbes LPPL/LPPLS calibrées.

            Parameters
            ----------
            fig, ax        : figure & axe retournés par _base()
            dict_results   : même structure que précédemment (résultats JSON)
            nb_calib       : nombre de calibrations aléatoires à afficher
            window_ext     : prolongation (en jours) après t2 pour la prédiction
            """
            import random
            from GQLib.Models import LPPL, LPPLS  # assure-toi que l'import reflète ton arborescence

            # 1) Récupère le set de calibrations (on suppose la clé « Set 1 » et « NELDER_MEAD »)
            try:
                calib_set = dict_results["Set 1"]["NELDER_MEAD"]["raw_run_result"]
            except (KeyError, TypeError):
                logger.error("dict_results ne contient pas la clé attendue « Set 1 → NELDER_MEAD → raw_run_result »")
                return

            if len(calib_set) == 0:
                logger.warning("Aucune calibration LPPL(S) trouvée.")
                return

            # 2) Sélectionne quelques calibrations au hasard
            nb_calib = min(nb_calib, len(calib_set))
            selected = random.sample(calib_set, nb_calib)

            for info in selected:
                sub_start, sub_end = info["sub_start"], info["sub_end"]
                best_params = info["bestParams"]

                # --- extrait la portion de prix utilisée pour la calibration
                mask_train = [(sub_start <= t <= sub_end) for t in self.global_times]
                y_train = np.array([p for p, m in zip(self.global_prices, mask_train) if m])

                # modèle LPPL ou LPPLS ?
                model_cls = LPPLS if len(best_params) == 7 else LPPL

                # temps normalisés pour le fit
                t_train = np.linspace(sub_start, sub_end, len(y_train))

                # instancie le modèle
                model = model_cls(params=best_params, t=t_train, y=y_train)

                # --- prépare la fenêtre d'affichage étendue
                mask_pred = [(sub_start <= t <= sub_end + window_ext) for t in self.global_times]
                dates_pred = [d for d, m in zip(self.global_dates, mask_pred) if m]
                model.t = np.array([t for t, m in zip(self.global_times, mask_pred) if m])

                # --- trace la prédiction
                tc_date = self.global_dates[int(round(model.tc))]
                ax.plot(dates_pred,
                        model.predict(),
                        ls="--", lw=2,
                        label=f"{model_cls.__name__} (tc={tc_date.strftime('%d-%m-%Y')})")

            ax.legend(title="LPPL fits", loc="upper left", fontsize=10)
            fig.autofmt_xdate()
