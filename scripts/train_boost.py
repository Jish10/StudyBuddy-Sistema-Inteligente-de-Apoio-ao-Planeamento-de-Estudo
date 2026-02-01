import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

from studybuddy.features import default_features, prepare, TARGET
from studybuddy.io import save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="StudentPerformanceFactors.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--min_samples_leaf", type=int, default=40)
    ap.add_argument("--early_stop", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    feats = default_features(df)
    X, y, cleaned = prepare(df, feats, args.target)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    mono = [1 if f in ("Hours_Studied", "Attendance", "Previous_Scores") else 0 for f in feats]

    model = HistGradientBoostingRegressor(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        l2_regularization=args.l2,
        min_samples_leaf=args.min_samples_leaf,
        early_stopping=args.early_stop,
        random_state=42,
        monotonic_cst=mono,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_val, pred))),
        "r2": float(r2_score(y_val, pred)),
        "features": feats,
        "monotonic_cst": mono,
        "params": {
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "l2_regularization": args.l2,
            "min_samples_leaf": args.min_samples_leaf,
            "early_stopping": bool(args.early_stop),
        }
    }

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feats, "rmse": metrics["rmse"]}, out/"predictor.joblib")
    save_json(metrics, out/"meta.json")

    print("✅ saved:", out/"predictor.joblib")
    print("✅ metrics:", metrics)

if __name__ == "__main__":
    main()
