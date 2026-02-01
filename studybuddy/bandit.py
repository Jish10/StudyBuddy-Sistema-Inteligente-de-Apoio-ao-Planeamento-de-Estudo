from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import math
from .model import Predictor

@dataclass
class BanditConfig:
    ucb_c: float = 1.8
    max_hours_per_subject: int = 14
    difficulty_weight: float = 0.25
    priority_weight: float = 0.35

def _defaults(predictor: Predictor) -> Dict[str, float]:
    d = {}
    for f in predictor.features:
        if f in ("Hours_Studied","Attendance","Previous_Scores"):
            continue
        if f == "Sleep_Hours":
            d[f] = 7.0
        elif f == "Tutoring_Sessions":
            d[f] = 0.0
        elif f == "Physical_Activity":
            d[f] = 3.0
        else:
            d[f] = 0.0
    return d

def allocate_hours_ucb(
    predictor: Predictor,
    subjects: List[str],
    total_hours: int,
    attendance: float,
    last_scores: Dict[str, float],
    difficulty: Dict[str, int],
    priority: Dict[str, int],
    cfg: BanditConfig,
) -> Dict[str, int]:
    alloc = {s: 0 for s in subjects}
    pulls = {s: 0 for s in subjects}

    defaults = _defaults(predictor)

    def boost(s: str) -> float:
        d = int(difficulty.get(s, 3))
        p = int(priority.get(s, 3))
        b = 1.0 + cfg.difficulty_weight * ((d - 3) / 2.0) + cfg.priority_weight * ((p - 3) / 2.0)
        return max(0.2, b)

    for t in range(1, int(total_hours)+1):
        best_s, best_ucb = None, -1e18
        for s in subjects:
            h = alloc[s]
            if h >= cfg.max_hours_per_subject:
                continue

            cur = float(last_scores.get(s, 60.0))
            kwargs = {"Attendance": float(attendance), **defaults}

            imp_now = predictor.improvement(cur_score=cur, hours_week=h, **kwargs)
            imp_next = predictor.improvement(cur_score=cur, hours_week=h+1, **kwargs)
            marginal = imp_next - imp_now

            sigma = predictor.uncertainty(h+1)
            ucb = (marginal * boost(s)) + cfg.ucb_c * sigma * math.sqrt(math.log(t + 1.0) / (pulls[s] + 1.0))

            if ucb > best_ucb:
                best_ucb, best_s = ucb, s

        if best_s is None:
            break
        alloc[best_s] += 1
        pulls[best_s] += 1

    return alloc
