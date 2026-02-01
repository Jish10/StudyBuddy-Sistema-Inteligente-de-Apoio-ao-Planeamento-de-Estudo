from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

from .model import Predictor
from .bandit import BanditConfig, allocate_hours_ucb, _defaults

@dataclass
class PlanConfig:
    weeks: int = 6
    total_hours_week: int = 18
    max_hours_per_subject: int = 14
    ucb_c: float = 1.8
    difficulty_weight: float = 0.25
    priority_weight: float = 0.35

def make_week_plan(
    predictor: Predictor,
    subjects: List[str],
    attendance: float,
    last_scores: Dict[str, float],
    difficulty: Dict[str, int],
    priority: Dict[str, int],
    cfg: PlanConfig,
) -> Dict[str, int]:
    bcfg = BanditConfig(
        ucb_c=cfg.ucb_c,
        max_hours_per_subject=cfg.max_hours_per_subject,
        difficulty_weight=cfg.difficulty_weight,
        priority_weight=cfg.priority_weight,
    )
    return allocate_hours_ucb(predictor, subjects, cfg.total_hours_week, attendance, last_scores, difficulty, priority, bcfg)

def simulate_weeks(
    predictor: Predictor,
    subjects: List[str],
    attendance: float,
    last_scores: Dict[str, float],
    difficulty: Dict[str, int],
    priority: Dict[str, int],
    cfg: PlanConfig,
):
    plans = []
    scores = {s: [float(last_scores.get(s, 60.0))] for s in subjects}
    cur = {s: float(last_scores.get(s, 60.0)) for s in subjects}
    defaults = _defaults(predictor)

    for _ in range(int(cfg.weeks)):
        alloc = make_week_plan(predictor, subjects, attendance, cur, difficulty, priority, cfg)
        plans.append(alloc)
        for s in subjects:
            h = float(alloc.get(s, 0))
            nxt = predictor.predict(Hours_Studied=h, Attendance=float(attendance), Previous_Scores=float(cur[s]), **defaults)
            cur[s] = float(0.65 * cur[s] + 0.35 * nxt)
            scores[s].append(cur[s])
    return plans, scores
