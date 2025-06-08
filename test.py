from GQLib.Optimizers.MLNN import MLNN
from GQLib.Optimizers.RNN import RNN
from GQLib.Optimizers.CNN import CNN
from GQLib.Models import LPPLS
from GQLib.enums import InputType
from GQLib.AssetProcessor import AssetProcessor
from GQLib.subintervals import ClassicSubIntervals, DidierSubIntervals
from torch import nn
import torch


# Classic Version
wti = AssetProcessor(input_type = InputType.WTI)

wti.compare_optimizers(frequency="daily",
                       optimizers=[CNN(LPPLS), RNN(LPPLS), MLNN(LPPLS)],
                       significativity_tc=0.3,
                       rerun=False,
                       nb_tc=10,
                       subinterval_method=ClassicSubIntervals,
                       save=False,
                       plot=True)


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
