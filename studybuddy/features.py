from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd

CANDIDATE_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Sleep_Hours",
    "Tutoring_Sessions",
    "Physical_Activity",
]

TARGET = "Exam_Score"

@dataclass
class Meta:
    features: List[str]
    target: str
    dataset_rows: int
    used_rows: int
    metrics: dict
    model_kind: str

def prepare(df: pd.DataFrame, features: List[str], target: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)[:60]}")
    work = df[features + [target]].copy()
    for c in features + [target]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna()
    X = work[features].to_numpy(dtype=np.float32)
    y = work[target].to_numpy(dtype=np.float32)
    return X, y, work

def default_features(df: pd.DataFrame) -> List[str]:
    feats = [c for c in CANDIDATE_FEATURES if c in df.columns]
    if "Hours_Studied" not in feats or "Attendance" not in feats or "Previous_Scores" not in feats:
        base = [c for c in ["Hours_Studied","Attendance","Previous_Scores"] if c in df.columns]
        if len(base) < 3:
            raise ValueError("Dataset doesn't contain required columns: Hours_Studied, Attendance, Previous_Scores")
        feats = base
    return feats
