from datetime import datetime
import json
from .Optimizers import MPGA, PSO, SGA, SA
from .Framework import Framework
from GQLib.Optimizers import Optimizer
from .enums import InputType
from typing import List, Dict, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

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
        # On load la config de notre input_type 
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

    def visualise_tc(self,
                           frequency : str = "daily",
                           optimizers : list[Optimizer] =  [SA(), SGA(), PSO(), MPGA()], 
                           rerun : bool = False,
                           nb_tc : int = None, 
                           significativity_tc = 0.3,
                           save : bool = False):
        
        """
        Visualizes the turning points (TC) for a given frequency and optimization method.
        The list of dates for each data is defined in the config file

        Args:
            frequency (str): The frequency of the data ('daily', 'weekly', or 'monthly').
            optimizers (list[Optimizer]): A list of optimizers to use for the analysis.
            rerun (bool): Whether to rerun the optimization process (default is False).
            nb_tc (int): The number of turning points to visualize (default is None, meaning all).
            significativity_tc (float): The significance threshold for the turning points (default is 0.3).
            save (bool): Whether to save the results as JSON files (default is False).
        """
    
        if frequency not in ["daily", "weekly", "monthly"]:
                raise ValueError("The frequency must be one of 'daily', 'weekly', 'monthly'.")
        
        #Initialisation du Framework
        fw = Framework(frequency = frequency, input_type=self.input_type)
        logging.info(f"FREQUENCY : {frequency}")

        # Visualisation pour chaque algorithme
        for optimizer in optimizers:
            current = 0
            logging.info(f"\nRunning process for {optimizer.__class__.__name__}")
            for set_name, (start_date, end_date) in self.dates_sets.items():

                graph_start_date, graph_end_date = self.dates_graphs[current]
                # Conversion des chaînes de dates en objets datetime pour faciliter le formatage
                start_date_obj = datetime.strptime(start_date, "%d/%m/%Y")
                end_date_obj = datetime.strptime(end_date, "%d/%m/%Y")
                filename = f"Results/results_{self.input_type.value}/{optimizer.__class__.__name__}/{frequency}/{optimizer.lppl_model.__name__}_{start_date_obj.strftime('%m-%Y')}_{end_date_obj.strftime('%m-%Y')}.json"
                
                if rerun : 
                    logging.info(f"Running process for {set_name} from {start_date} to {end_date}")

                    # Exécute le processus d'optimisation pour l'intervalle de dates donné
                    results = fw.process(start_date, end_date, optimizer)
                    if save:
                        # Sauvegarde des résultats au format JSON dans le fichier généré
                        fw.save_results(results, filename)
                    # Verification de la significativité des résultats
                    best_results = fw.analyze(results, significativity_tc=significativity_tc, lppl_model = optimizer.lppl_model)

                else:
                    best_results = fw.analyze(result_json_name=filename,significativity_tc=significativity_tc,lppl_model=optimizer.lppl_model)
                
                # Visualisation des résultats finaux
                fw.visualize_tc(
                    best_results,
                    f"{self.input_type.value} {optimizer.__class__.__name__} {frequency} ({optimizer.lppl_model.__name__}) results from {start_date_obj.strftime('%m-%Y')} to {end_date_obj.strftime('%m-%Y')}",
                    start_date=graph_start_date,
                    end_date=graph_end_date,
                    nb_tc = nb_tc,
                    real_tc = self.real_tcs[current]
                )
                current+=1

    def old_compare_optimizers(self,
                               frequency : str = "daily",
                               optimizers: list[Optimizer] = [SA(), SGA(), PSO(), MPGA()],
                               significativity_tc=0.3,
                               nb_tc : int = 20,
                               rerun: bool = False,
                               save: bool = False,
                               save_plot : bool = False):
    
        """
        Compares the performance of different optimizers in predicting turning points.

        Args:
            frequency (str): The frequency of the data ('daily', 'weekly', or 'monthly').
            optimizers (list[Optimizer]): A list of optimizers to compare.
            significativity_tc (float): The significance threshold for the turning points (default is 0.3).
            nb_tc (int): The number of turning points to visualize (default is 20).
            rerun (bool): Whether to rerun the optimization process (default is False).
            save (bool): Whether to save the results as JSON files (default is False).
            save_plot (bool): Whether to save the comparison plot (default is False).
        """
        if frequency not in ["daily", "weekly", "monthly"]:
                raise ValueError("The frequency must be one of 'daily', 'weekly', 'monthly'.")
        
        fw = Framework(frequency = frequency, input_type=self.input_type)
        
        logging.info(f"FREQUENCY : {frequency}")
        compteur = 0

        for set_name, (start_date, end_date) in self.dates_sets.items():
            logging.info(f"Running process for {set_name} from {start_date} to {end_date}")
            best_results_list = {}
            optimiseurs_models = []

            for optimizer in optimizers:
                optimiseurs_models.append(optimizer.lppl_model.__name__)
                # Conversion des chaînes de dates en objets datetime pour faciliter le formatage
                start_date_obj = datetime.strptime(start_date, "%d/%m/%Y")
                end_date_obj = datetime.strptime(end_date, "%d/%m/%Y")
                filename = f"Results/results_{self.input_type.value}/{optimizer.__class__.__name__}/{frequency}/{ optimizer.lppl_model.__name__}_{start_date_obj.strftime('%m-%Y')}_{end_date_obj.strftime('%m-%Y')}.json"
                
                if rerun:
                    logging.info(f"\nRunning process for {optimizer.__class__.__name__}")
                    results = fw.process(start_date, end_date, optimizer)
                    best_results_list[optimizer.__class__.__name__] = fw.analyze(results=results,
                                                                                    significativity_tc=significativity_tc,
                                                                                    lppl_model=optimizer.lppl_model)
                    if save:
                        fw.save_results(results, filename)
                else:
                    logging.info(f"Getting result for {optimizer.__class__.__name__}\n")
                    best_results_list[optimizer.__class__.__name__] = fw.analyze(result_json_name=filename,
                                                                                    significativity_tc=significativity_tc,
                                                                                    lppl_model=optimizer.lppl_model)
                    
            real_tc = self.real_tcs[compteur] if len(self.real_tcs)>compteur else None
            fw.visualize_compare_results(multiple_results=best_results_list, 
                                        name=f"Predicted critical times {frequency} {self.input_type.value} from {start_date_obj.strftime('%m-%Y')} to {end_date_obj.strftime('%m-%Y')}",
                                        data_name=f"{self.input_type.value} Data", 
                                        real_tc=real_tc, 
                                        optimiseurs_models = optimiseurs_models,
                                        start_date=start_date,
                                        end_date=end_date,
                                        nb_tc = nb_tc,
                                        save_plot = save_plot)
            compteur += 1

    def visualise_data(self,
                       frequency : str = "daily",
                       start_date = None,
                        end_date = None):
        """
        Visualizes the raw data for a given frequency and date range.
        
        Args:
            frequency (str): The frequency of the data ('daily', 'weekly', or 'monthly').
            start_date (str): The start date for the visualization (default is None).
            end_date (str): The end date for the visualization (default is None).
        """

        if frequency not in ["daily", "weekly", "monthly"]:
                raise ValueError("The frequency must be one of 'daily', 'weekly', 'monthly'.")
        
        fw = Framework(frequency = frequency, input_type=self.input_type)
        fw.visualise_data(start_date, end_date)

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

                    optim_results[optimizer.__class__.__name__] = self.run_optimizer(fw, start_date, end_date, optimizer, self.real_tcs[idx])

            results[set_name] = optim_results

        with open(f"Venise_Results/{self.input_type.value}_{frequency}.json", "w") as file:
            json.dump(results, file, indent=4)

    def run_optimizer(self, fw: Framework, start_date: datetime, end_date: datetime, optimizer: Optimizer, real_tc: float) -> Dict:
        """
        Run the optimizer on the data, filter the resulting turning points, analyse and save the results.

        Args:
            fw (Framework): The insttanciated Framework object with the data.
            start_date (datetime): The start date for the optimization.
            end_date (datetime): The end date for the optimization.
            optimizer (Optimizer): The optimizer to use for the analysis.
            real_tc (float): The real turning point to compare with.

        Returns:
            Dict: A dictionary containing the information about the turning points distribution.
        """
        run_results = fw.process(start_date, end_date, optimizer)
        filtered_results = fw.analyze(results=run_results, lppl_model=optimizer.lppl_model, show=True)

        tc_info = self._get_tc_distrib(filtered_results, real_tc)
        tc_info["Confidence"] = self._compute_confidence(run_results, filtered_results)

        return tc_info
    
    @staticmethod
    def _get_tc_distrib(self, run_results: List[Dict], real_tc: float) -> Dict:
        """
        Get the distribution of turning points (TC) from the optimization results.
        Compute quantiles, mean, and standard deviation of the turning points.

        Args:
            run_results (List[Dict]): A list of dictionaries containing the optimization results.

        Returns:
            Dict: A dictionary containing the info on the turning points distribution.
        """
        tc_distrib = [tc for tc in run_results["bestParams"][0]]

        return {
            "distrib": sorted(tc_distrib),
            "mean_error": self._compute_mean_error(tc_distrib, real_tc),
            "mean": np.mean(tc_distrib),
            "std": np.std(tc_distrib),
            "quantiles": {
                "25": np.quantile(tc_distrib, 0.25),
                "50": np.quantile(tc_distrib, 0.5),
                "75": np.quantile(tc_distrib, 0.75)
            }
        }

    @staticmethod
    def _compute_confidence(self, run_results: Dict, filtered_results: Dict) -> float:
        """
        Compute the confidence of the turning points distribution by and the chosen filtering method.
        Confidence = (Number of significant turning points) / (Total number of turning points)

        Args:
            run_results (Dict): The results of the optimization process.
            filtered_results (Dict): The filtered results after analysis.

        Returns:
            float: The confidence of the turning points distribution.
        """
        return len([f["bestParams"][0] for f in filtered_results if f["is_significant"] is True]) / len(run_results["bestParams"][0])
    
    @staticmethod
    def _compute_mean_error(self, tc_distrib: List[float], real_tc: float) -> float:
        """
        Compute the mean error of the predicted turning points.

        Args:
            tc_distrib (List[float]): A list of turning points.

        Returns:
            float: The mean error of the turning points distribution.
        """
        return np.mean([abs(tc - np.mean(tc_distrib)) for tc in tc_distrib])