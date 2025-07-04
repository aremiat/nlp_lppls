from enum import Enum
from GQLib.Models import LPPL, LPPLS
from GQLib.Optimizers import NELDER_MEAD

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
    NELDER_MEAD = NELDER_MEAD
