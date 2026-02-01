from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

def clamp(x: float, lo=0.0, hi=100.0) -> float:
    return float(max(lo, min(hi, x)))

@dataclass
class Predictor:
    model: object
    features: List[str]
    rmse_val: float

    def predict(self, **kwargs) -> float:
        x = np.array([[float(kwargs[f]) for f in self.features]], dtype=np.float32)
        y = float(self.model.predict(x)[0])
        return clamp(y)

    def predict_curve(self, hours_feature: str, hours_max: int, **kwargs):
        xs = list(range(0, int(hours_max)+1))
        ys = []
        for h in xs:
            kwargs2 = dict(kwargs)
            kwargs2[hours_feature] = float(h)
            ys.append(self.predict(**kwargs2))
        best = -1e18
        ys_m = []
        for y in ys:
            best = max(best, y)
            ys_m.append(best)
        return xs, ys, ys_m

    def improvement(self, cur_score: float, hours_week: float, **kwargs) -> float:
        nxt = self.predict(**{**kwargs, "Hours_Studied": hours_week, "Previous_Scores": cur_score})
        return float(nxt - float(cur_score))

    def uncertainty(self, hours_week: float) -> float:
        return float(self.rmse_val / np.sqrt(1.0 + float(hours_week)))
