import matplotlib.pyplot as plt
import json
import pandas as pd
import os

from GQLib.Framework import Framework
from GQLib.enums import InputType
from GQLib.Plotters.kernel_plotter import KernelPlotter
from GQLib.Optimizers.NELDER_MEAD import NELDER_MEAD
from GQLib.Optimizers.Neural_Network.MLNN import MLNN
from GQLib.Optimizers.Neural_Network.RNN import RNN
from GQLib.Optimizers.Neural_Network.CNN import CNN
from GQLib.Optimizers.Neural_Network.attention import MLNNWithAttention, CNNWithAttention, RNNWithAttention
from GQLib.Models.LPPLS import LPPLS
from GQLib.subintervals import ClassicSubIntervals
from GQLib.AssetProcessor import AssetProcessor



if __name__ == "__main__":

    ####################################################################################################################
    ############################################# RUN THE RESULT WITHOUT SELF-ATTENTION ################################
    ####################################################################################################################


    wti = AssetProcessor(input_type = InputType.WTI)
    wti.compare_optimizers(frequency="daily",
                           optimizers=[NELDER_MEAD(LPPLS), # Optimizers
                               MLNN(LPPLS, epochs=100, silent=True),
                                        RNN(LPPLS, epochs=100, silent=True),
                                        CNN(LPPLS, epochs=100, silent=True)],
                           significativity_tc=0.3,
                           rerun=False, # set rerun to True if you want to re-run the optimizers
                           nb_tc=10, # the number of tc use to make the plot
                           subinterval_method=ClassicSubIntervals,
                           save=False, # set save to True if you want to save the results
                           plot=False) # set plot to True if you want to plot the results


    ####################################################################################################################
    ################################################ DENSITY PLOT ######################################################
    ####################################################################################################################


    fw = Framework("daily", InputType.WTI)
    plotter = KernelPlotter.from_framework(fw)
    fig, axes = plotter._base("01/04/2003", "02/01/2008", "03/07/2008")
    with open("nlp_results/WTI_metrics.json", "r") as f:
        data = json.load(f)

    plotter._add_half_violins(fig, axes, data["Set 1"],
                            width_scale=0.5,
                            spacing=0.75,
                            specific="tc_distrib_non_significant",
                            hatch_pattern="\\",
                            color="white",
                            text=True,)
    plotter._add_half_violins(fig, axes, data["Set 1"],
                            width_scale=0.5,
                            spacing=0.75,
                            specific="tc_distrib_significant",
                            hatch_pattern="//",
                            color="lightgrey",)
    plotter._add_half_violins(fig, axes, data["Set 1"],
                            width_scale=0.5,
                            spacing=0.75,
                            specific="tc_distrib_significant_power",
                            hatch_pattern="x",
                            color="grey",)

    plt.show(block=True)


    ####################################################################################################################
    ########################################## RUN THE RESULT WITH SELF-ATTENTION ######################################
    ####################################################################################################################


    wti = AssetProcessor(input_type = InputType.WTI)
    wti.compare_optimizers(frequency="daily",
                           optimizers=[NELDER_MEAD(LPPLS),
                               MLNN(LPPLS, net=MLNNWithAttention(), epochs=100, silent=True),
                                        RNN(LPPLS, net=RNNWithAttention(), epochs=100, silent=True),
                                        CNN(LPPLS, net=CNNWithAttention(), epochs=100, silent=True)],
                           significativity_tc=0.3,
                           rerun=False, # set rerun to True if you want to re-run the optimizers
                           nb_tc=10, # the number of tc use to make the plot
                           subinterval_method=ClassicSubIntervals,
                           save=False, # set save to True if you want to save the results
                           plot=False) # set plot to True if you want to plot the results


    ####################################################################################################################
    ################################################ Save the results ##################################################
    ####################################################################################################################


    out_dir = "nlp_results"
    symbols = ["WTI", "BTC", "SP500"]

    for sym in symbols:
        json_file = f"nlp_results/{sym}_metrics.json"
        if not os.path.exists(json_file):
            print(f"⚠️ File {json_file} not found, skipping.")
            continue

        with open(json_file, "r") as f:
            metrics = json.load(f)

        for set_name in ("Set 1", "Set 2", "Set 3"):
            if set_name not in metrics:
                continue

            rows = []
            for algo, d in metrics[set_name].items():
                rows.append(
                    {
                        "algorithm": algo,
                        "tc_power_mean": d.get("tc_power_mean"),
                        "confidence": d.get("confidence"),
                        "error_mean": d.get("error_mean"),
                        "error_std": d.get("error_std"),
                    }
                )
            df = pd.DataFrame(rows)
            df["tc_power_mean"] = df["tc_power_mean"].round(1)
            df["confidence"] = df["confidence"].round(2)
            df["error_mean"] = df["error_mean"].round(1)
            df["error_std"] = df["error_std"].round(1)

            csv_fname = f"{sym}_{set_name.replace(' ', '')}_metrics.csv"
            csv_path = os.path.join(out_dir, csv_fname)
            print(f"Saving {csv_path}...")
            df.to_csv(csv_path, index=False)