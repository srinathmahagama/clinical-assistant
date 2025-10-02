# path: apps/MLService2/app/training/scripts/tune_xgb.py
from pathlib import Path
import argparse, json
import numpy as np, optuna, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

APP = Path(__file__).resolve().parents[2]  # .../MLService2/app
PROC = APP / "training" / "data" / "processed"
OUT_PATH = PROC / "best_xgb_params.json"

# -----------------------------
# Search space
# -----------------------------
def make_search_space(trial: optuna.Trial, mode: str):
    if mode == "fast":  # quick dev tuning
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 500),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 2.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            objective="multi:softprob", eval_metric="mlogloss",
            tree_method="hist", nthread=-1, verbosity=0,
        )
    # full mode
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 500, 1500),
        max_depth=trial.suggest_int("max_depth", 4, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 15),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        objective="multi:softprob", eval_metric="mlogloss",
        tree_method="hist", nthread=-1, verbosity=0,
    )

# -----------------------------
# Main tuning entry
# -----------------------------
def main(mode: str, n_trials: int, seed: int, cv: int):
    # Load data
    X = pd.read_parquet(PROC / "X.parquet")
    y = pd.read_parquet(PROC / "y.parquet")["severity"].astype(str)

    # Label encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Scale only numeric columns (vitals + age)
    numeric_cols = [c for c in X.columns if X[c].dtype in ["int64", "float64"]]
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    np.random.seed(seed)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    def objective(trial: optuna.Trial):
        params = make_search_space(trial, mode)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        f1s = []

        for tr, va in skf.split(X, y_enc):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y_enc[tr], y_enc[va]

            # compute balanced sample weights
            weights = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
            class_weight_map = {i: w for i, w in enumerate(weights)}
            w_tr = np.array([class_weight_map[i] for i in ytr], dtype=np.float32)

            model = XGBClassifier(**params)
            model.fit(
                Xtr, ytr,
                sample_weight=w_tr,
                eval_set=[(Xva, yva)],
                verbose=False
            )

            pred = model.predict(Xva)
            f1s.append(f1_score(yva, pred, average="macro"))

        return float(np.mean(f1s))

    # Run Optuna search
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save best params
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(study.best_params, f, indent=2)

    print("\n[tune_xgb] mode:", mode)
    print("[tune_xgb] best macro-F1:", study.best_value)
    print("[tune_xgb] best params saved to:", OUT_PATH)

# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Optuna tuner for XGBoost (multiclass).")
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=3, help="CV folds (3=fast, 5=stable)")
    args = ap.parse_args()
    main(mode=args.mode, n_trials=args.n_trials, seed=args.seed, cv=args.cv)
