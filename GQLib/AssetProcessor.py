from datetime import datetime, timedelta
import json
from .Optimizers import MPGA, PSO, SGA, SA
from .Framework import Framework
from GQLib.Optimizers import Optimizer
from .enums import InputType
from typing import List, Dict, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)
from GQLib.logging import with_spinner

class AssetProcessor:
    """
    This class processes financial asset data, applies optimization algorithms on LPPL Models,
    and visualizes the results based on various configurations.
    """
    def __init__(self, input_type : InputType = InputType.WTI, rerun : bool = False):
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
                           save_plot : bool = False) -> None:
        """
        Compare the performance of different optimizers on the same data set over multiple date ranges.

        Args:
            optimizers (list[Optimizer]): A list of optimizers to compare.
            frequency (str): The frequency of the data ('daily', 'weekly', or 'monthly').
        """
        fw = Framework(frequency = frequency, input_type=self.input_type)
        
        results = {}
        for idx, (set_name, (start_date, end_date)) in enumerate(self.dates_sets.items()):
            logging.info(f"Running process for {set_name} from {start_date} to {end_date}")

            optim_results = {}
            for optimizer in optimizers:
                    logging.info(f"\nRunning process for {optimizer.__class__.__name__}")

                    optim_results[optimizer.__class__.__name__] = self.run_optimizer(fw, start_date, end_date, optimizer, self.real_tcs[idx], rerun)

            results[set_name] = optim_results

        with open(f"Venise_Results/{self.input_type.value}_metrics.json", "w") as file:
            json.dump(results, file, indent=4)
        logging.info(f"Results saved to Venise_Results/{self.input_type.value}_{frequency}.json")

    def run_optimizer(self, fw: Framework, start_date: datetime, end_date: datetime, optimizer: Optimizer, real_tc: float, rerun: bool = False) -> Dict:
        """
        Run the optimizer on the data, filter the resulting turning points, analyse and save the results.

        Args:
            fw (Framework): The insttanciated Framework object with the data.
            start_date (datetime): The start date for the optimization.
            end_date (datetime): The end date for the optimization.
            optimizer (Optimizer): The optimizer to use for the analysis.
            real_tc (str): The real turning point to compare with.

        Returns:
            Dict: A dictionary containing the information about the turning points distribution.
        """
        real_tc_numeric = self._translate_tc_to_numeric(real_tc, fw)

        if rerun:
            run_results = fw.process(start_date, end_date, optimizer)
            filtered_results = fw.analyze(results=run_results, lppl_model=optimizer.lppl_model, show=True)
        else:
            start_date_obj = datetime.strptime(start_date, "%d/%m/%Y")
            end_date_obj = datetime.strptime(end_date, "%d/%m/%Y")
            filename = f"Results/results_{self.input_type.value}/{optimizer.__class__.__name__}/daily/{optimizer.lppl_model.__name__}_{start_date_obj.strftime('%m-%Y')}_{end_date_obj.strftime('%m-%Y')}.json"
            with open(filename, "r") as f:
                run_results = json.load(f)
            filtered_results = fw.analyze(result_json_name=filename, lppl_model=optimizer.lppl_model)
        
        forward_end_date = datetime.strftime(datetime.strptime(end_date, "%d/%m/%Y") + timedelta(days=300), "%d/%m/%Y")
        fw.visualize_tc(filtered_results, "Test", "WTI", start_date, forward_end_date, real_tc=real_tc)

        return self._compute_tc_metrics(run_results, filtered_results, real_tc_numeric, fw.data[-1, 0])
    
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

        return {
            "tc_distrib": sorted(all_tc),
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
            }
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