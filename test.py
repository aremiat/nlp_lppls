import matplotlib.pyplot as plt
from GQLib.Framework import Framework
from GQLib.enums import InputType
import json
from GQLib.Plotters.kernel_plotter import KernelPlotter
from GQLib.Optimizers.Neural_Network.MLNN import MLNN
from GQLib.Optimizers.Neural_Network.RNN import RNN
from GQLib.Optimizers.Neural_Network.CNN import CNN
from GQLib.Optimizers.Neural_Network.NLCNN import NLCNN
from GQLib.Models.LPPLS import LPPLS
from GQLib.subintervals import ClassicSubIntervals
from GQLib.AssetProcessor import AssetProcessor

# Classic Version
wti = AssetProcessor(input_type = InputType.WTI)

wti.compare_optimizers(frequency="daily",
                       optimizers=[MLNN(LPPLS)],
                       significativity_tc=0.3,
                       rerun=True,
                       nb_tc=10,
                       subinterval_method=ClassicSubIntervals,
                       save=True,
                       plot=False)

# PLotting the results

fw = Framework("daily", InputType.WTI)

plotter = KernelPlotter.from_framework(fw)

fig, axes = plotter._base("01/04/2003", "02/01/2008", "03/07/2008")
#
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

fw = Framework("daily", InputType.WTI)

fig, axes = plotter._base("01/02/2007", "01/02/2011", "29/04/2011")

#
with open("nlp_results/WTI_metrics.json", "r") as f:
    data = json.load(f)

plotter._add_half_violins(fig, axes, data["Set 2"],
                        width_scale=0.5,
                        spacing=0.75,
                        specific="tc_distrib_non_significant",
                        hatch_pattern="\\",
                        color="white",
                        text=True,)

plt.show(block=True)


fig, axes = plotter._base("01/11/2011", "31/12/2016", "11/02/2016")

#
with open("nlp_results/WTI_metrics.json", "r") as f:
    data = json.load(f)

plotter._add_half_violins(fig, axes, data["Set 3"],
                        width_scale=0.5,
                        spacing=0.75,
                        specific="tc_distrib_non_significant",
                        hatch_pattern="\\",
                        color="white",
                        text=True,)

plt.show(block=True)

# Custom Version

# Define a custom neural network
# class CustomNet(nn.Module):
#     def __init__(self, n_hidden: int = 64):
#         super().__init__()
#         self.fc1 = nn.Linear(1, n_hidden)
#         self.fc2 = nn.Linear(n_hidden, n_hidden)
#         self.out = nn.Linear(n_hidden, 3)
#         self.relu = nn.ReLU()
#
#     def forward(self, t: torch.Tensor) -> torch.Tensor:
#         h = self.relu(self.fc1(t))
#         h = self.relu(self.fc2(h))
#         raw = self.out(h)
#         return raw.mean(dim=0)
#
#
# # Instantiate the custom network
# custom_net = CustomNet(n_hidden=64)
#
# # Use the custom network in MLNN
# wti.compare_optimizers(frequency="daily",
#                        optimizers=[RNN(LPPLS, net=custom_net)],
#                        significativity_tc=0.3,
#                        rerun=True,
#                        nb_tc=10,
#                        subinterval_method=ClassicSubIntervals,
#                        save=True,
#                        plot=True)
