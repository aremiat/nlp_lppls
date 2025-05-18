from enum import Enum
from GQLib.Models import LPPL, LPPLS
from GQLib.Optimizers import NELDER_MEAD, MPGA, PSO, SA, SGA, TABU, FA

class InputType(Enum):
    WTI = "WTI"
    SP500 = "SP500"
    BTC = "BTC"
    USO = "USO"
    SSE = "SSE"
    EURUSD = "EURUSD"

class Models(Enum):
    LPPL = LPPL
    LPPLS = LPPLS

class Optimizers(Enum):
    MPGA = MPGA
    PSO = PSO
    SA = SA
    SGA = SGA
    NELDER_MEAD = NELDER_MEAD
    TABU = TABU
    FA = FA
