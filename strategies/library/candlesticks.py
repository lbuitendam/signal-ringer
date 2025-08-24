from __future__ import annotations
from typing import List
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from engine.utils import Signal

# … keep your detect_* helpers …

class CandlesStrategy(BaseStrategy):
    name = "Candlestick Patterns"
    CATEGORY = "Candlestick"
    PARAMS_SCHEMA = {
        "which": {
            "type": "select",
            "options": [
                "hammer","inv_hammer","hanging_man","shooting_star",
                "engulf_bull","engulf_bear","tweezer_top","tweezer_bottom",
            ],
            "default": "hammer",
            "label": "Pattern",
        }
    }

    def __init__(self, which: str = "hammer", params=None):
        super().__init__(params or {})
        self.which = (params or {}).get("which", which)
    # … keep your signals() body unchanged …
