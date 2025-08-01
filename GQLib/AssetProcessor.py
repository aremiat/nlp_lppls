from datetime import datetime, timedelta
import json
from .subintervals import ClassicSubIntervals, DidierSubIntervals, SubIntervalMethod
from .Framework import Framework
from GQLib.Optimizers import Optimizer
from .enums import InputType
from typing import List, Dict
import logging
import numpy as np
from GQLib.logging import with_spinner

logger = logging.getLogger(__name__)

with open("params/debug_framework.json", "r") as file:
    debug_params = json.load(file)

DEBUG_STATUS_GRAPH_TC = debug_params["DEBUG_STATUS_GRAPH_TC"]
DEBUG_STATUS_GRAPH_LOMB = debug_params["DEBUG_STATUS_GRAPH_LOMB"]
DEBUG_STATUS_GRAPH_COMPARE_RECTANGLE = debug_params["DEBUG_STATUS_GRAPH_COMPARE_RECTANGLE"]
DEBUG_STATUS_GRAPH_LPPL_FIT = debug_params["DEBUG_STATUS_GRAPH_LPPL_FIT"]

with open("params/lomb.json", "r") as file:
    lomb_params = json.load(file)

QUANTILE_STATUS = lomb_params["QUANTILE_STATUS"]
NB_TC = lomb_params["NB_TC"]

class AssetProcessor:
    """
    This class processes financial asset data, applies optimization algorithms on LPPL Models,
    and visualizes the results based on various configurations.
    """
    def __init__(self, input_type : InputType = InputType.WTI):
        """
        Initializes the AssetProcessor with a specified input type (e.g., WTI).
        
        Args:
            input_type (InputType): The type of asset data to process (default is WTI).
        """
        self.input_type = input_type
        logging.info(f"Input type: {self.input_type}")
        config = self.load_config()
        self.dates_sets = config["sets"]
        self.dates_graphs = config["graphs"]
        self.real_tcs = config["real_tcs"]


    def load_config(self):
        """
        Loads the configuration for the specified input type from a JSON file.
        
        Returns:
            dict: A dictionary containing the configuration data for the specified input type.
        
        Raises:
            ValueError: If the input type is not found in the configuration file.
        """
        with open("params/config.json", "r") as file:
            config = json.load(file)
        if self.input_type.name not in config:
            raise ValueError(f"Input type {self.input_type.name} not found in configuration.")
        return config[self.input_type.name]

    # ---------------------------------------------
    # |                 NEW VERSION               |
    # ---------------------------------------------
    def compare_optimizers(self, optimizers: list[Optimizer], 
                           frequency : str = "daily",
                           significativity_tc=0.3,
                           nb_tc : int = 20,
                           rerun: bool = False,
                           save: bool = False,
                           subinterval_method = None,
                           plot: bool = False) -> None:
        """
        Compare the performance of different optimizers on the same data set over multiple date ranges.

        Args:
            optimizers (list[Optimizer]): A list of optimizers to compare.
            frequency (str): The frequency of the data ('daily', 'weekly', or 'monthly').
        """

        lomb_params = {}
        lomb_params["significativity_threshold"] = significativity_tc
        lomb_params["nb_tc"] = nb_tc

        if subinterval_method is None:
            subinterval_method = ClassicSubIntervals


        fw = Framework(frequency = frequency, input_type=self.input_type, subinterval_method=subinterval_method)
        
        results = {}
        for idx, (set_name, (start_date, end_date)) in enumerate(self.dates_sets.items()):
            
            logging.info(f"Running process for {set_name} from {start_date} to {end_date}")

            optim_results = {}
            model_associated = []
            for optimizer in optimizers:
                    model_associated.append(optimizer.lppl_model.__name__)
                    logging.info(f"\nRunning process for {optimizer.__class__.__name__}")
                    optim_results[optimizer.__class__.__name__] = self.run_optimizer(fw, start_date, end_date, optimizer,
                                                                                     self.real_tcs[idx], rerun, save, plot)

            results[set_name] = optim_results

        class ResultEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (bool, np.bool_)):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

    
        with open(f"nlp_results/{self.input_type.value}_metrics.json", "w") as file:
            json.dump(results, file, indent=4, cls=ResultEncoder)
        logging.info(f"Results saved to nlp_results/{self.input_type.value}_metrics.json")

    def run_optimizer(
        self,
        fw: Framework,
        start_date: str,
        end_date: str,
        optimizer: Optimizer,
        real_tc: float,
        rerun: bool = False,
        save: bool = False,
        plot: bool = False
    ) -> Dict:
        """
        Run the optimizer on the data, filter the resulting turning points, analyse and save the results.

        Args:
            fw (Framework): The instantiated Framework object with the data.
            start_date (str): The start date for the optimization (format "dd/mm/YYYY").
            end_date (str): The end date for the optimization (format "dd/mm/YYYY").
            optimizer (Optimizer): The optimizer to use for the analysis.
            real_tc (float): The real turning point to compare with.
            rerun (bool, optional): If True, rerun the optimizer even if results exist on disk.
            save (bool, optional): If True, save the raw run_results to a JSON file.
            plot (bool, optional): If True, call Plotter.visualize_compare_results on the filtered results.

        Returns:
            Dict: A dictionary containing the information about the turning points distribution.
        """
        real_tc_numeric = self._translate_tc_to_numeric(real_tc, fw)
        start_date_obj = datetime.strptime(start_date, "%d/%m/%Y")
        end_date_obj = datetime.strptime(end_date, "%d/%m/%Y")

        filename = (
            f"Results/results_{self.input_type.value}/"
            f"{optimizer.__class__.__name__}/daily/"
            f"{optimizer.lppl_model.__name__}_"
            f"{start_date_obj.strftime('%m-%Y')}_"
            f"{end_date_obj.strftime('%m-%Y')}.json"
        )

        if rerun:
            run_results = fw.process(start_date, end_date, optimizer)
            filtered_results = fw.analyze(results=run_results, lppl_model=optimizer.lppl_model)
            if save:
                fw.save_results(run_results, filename)
        else:
            with open(filename, "r") as f:
                run_results = json.load(f)
            filtered_results = fw.analyze(result_json_name=filename, lppl_model=optimizer.lppl_model)

        # Si on doit tracer, on transforme la liste `filtered_results`
        # en un dict avec pour clé le nom de l'optimizer
        if plot:
            forward_end_date = datetime.strftime(
                datetime.strptime(end_date, "%d/%m/%Y") + timedelta(days=300),
                "%d/%m/%Y"
            )

            multiple_results_dict = {
                optimizer.__class__.__name__: {
                    "raw_filtered_result": filtered_results
                }
            }

            fw.plotter(
                "visualize_compare_results",
                multiple_results=multiple_results_dict,
                name=f"{optimizer.__class__.__name__} Results",
                data_name="WTI",
                start_date=start_date,
                end_date=forward_end_date,
                real_tc=real_tc
            )

        return self._compute_tc_metrics(
            run_results,
            filtered_results,
            real_tc_numeric,
            fw.data[-1, 0]
        )
    
    @with_spinner("Computing metrics...")
    def _compute_tc_metrics(self, run_results: List[Dict], filtered_results: List[Dict], real_tc: float, end_date: float) -> Dict:
        """
        Get the distribution of turning points (TC) from the optimization results.
        Compute quantiles, mean, and standard deviation of the turning points.

        Args:
            run_results (List[Dict]): Optimization results
            filtered_results (List[Dict]): Filtered results after analysis
            real_tc (float): The real turning point to compare with
            end_date (float): The end date for the optimization

        Returns:
            Dict: A dictionary containing the info on the turning points distribution.
        """
        # Rajouter le treshold dans le filtre
        all_tc = [tc["bestParams"][0] for tc in filtered_results]
        tc_errors = [(tc - real_tc) for tc in all_tc]

        # Trier les résultats par rapport à la valeur de leur power
        temp_tc_power = [
            [tc["bestParams"][0], tc["power_value"]]
            for tc in sorted(filtered_results, key=lambda tc: tc["power_value"])
            if tc["is_significant"] == np.True_
        ]
        temp_tc_power.sort(key=lambda x: x[1], reverse=True)

        if QUANTILE_STATUS:
            quantile = NB_TC / 100
            tc_significant_power = [tc for tc, power in temp_tc_power[:max(1, int(len(temp_tc_power) * quantile))]]
        else:
            tc_significant_power = [tc for tc, power in temp_tc_power[:NB_TC]]

        sorted([tc["bestParams"][0] for tc in filtered_results if tc["is_significant"] == np.True_])
        return {
            "tc_distrib": sorted(all_tc),
            "tc_distrib_significant": sorted([tc["bestParams"][0] for tc in filtered_results if tc["is_significant"] == np.True_]),
            "tc_distrib_non_significant": sorted([tc["bestParams"][0] for tc in filtered_results if tc["is_significant"] == np.False_]),
            "tc_distrib_significant_power": tc_significant_power,
            "tc_power_mean": np.mean(tc_significant_power),
            "confidence": len([f for f in filtered_results if f["is_significant"] == np.True_]) / len(run_results),
            "error_distrib": sorted(tc_errors),
            "error_mean": np.mean(tc_errors),
            "error_std": np.std(tc_errors),
            "quantiles": {
                "10": np.quantile(tc_errors, 0.1),
                "25": np.quantile(tc_errors, 0.25),
                "50": np.quantile(tc_errors, 0.5),
                "75": np.quantile(tc_errors, 0.75),
                "90": np.quantile(tc_errors, 0.9),
            },
            "raw_run_result": run_results,
            "raw_filtered_result": filtered_results,
            "real_tc": real_tc,
            "end_date": end_date,
        }

    def _translate_tc_to_numeric(self, tc: str, fw: Framework) -> float:
        """
        Convert the real turning point date to a numeric value based on the data range and the date bounds of the selected interval.

        Args:
            tc (str): The real turning point date in string format
            fw (Framework): The Framework object containing the data with the date range
        
        Returns:
            float: The numeric value of the turning point date.
        """
        tc_date = datetime.strptime(tc, "%d/%m/%Y")
        for i, date in enumerate(fw.data[:, 1]):
            if tc_date <= date:
                return fw.data[i, 0]
        raise ValueError(f"Turning point {tc} not found in the data range.")