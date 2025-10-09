# path: apps/MLService2/app/training/scripts/train_severity_lr_boosted.py
from pathlib import Path
import json, argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

import joblib

APP_ROOT = Path(__file__).resolve().parents[2]  # .../MLService2/app
PROC     = APP_ROOT / "training" / "data" / "processed"
MODELS   = APP_ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

def detect_continuous_columns(X: pd.DataFrame, meta: dict):
    """Prefer meta-specified candidates; fallback to numeric cols with >10 uniques."""
    meta_list = meta.get("continuous_candidates", [])
    cols = [c for c in meta_list if c in X.columns]
    if not cols:
        cols = [c for c in X.columns
                if np.issubdtype(X[c].dtype, np.number)
                and pd.Series(X[c]).nunique(dropna=True) > 10]
    return cols

def main(poly: bool, cv: int, random_state: int, max_iter: int):
    # Load data & meta
    X = pd.read_parquet(PROC / "X.parquet")
    y = pd.read_parquet(PROC / "y.parquet")["severity"].astype(str)
    with open(PROC / "meta.json") as f:
        meta = json.load(f)

    # Keep only three valid labels
    valid = y.isin({"Mild", "Moderate", "Severe"})
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.20, random_state=random_state, stratify=y_enc
    )

    # Columns to scale (continuous only)
    cont_cols = detect_continuous_columns(X_tr, meta)
    print(f"[lr] continuous columns to scale: {len(cont_cols)}")

    # Preprocessor
    transformers = []
    if poly and cont_cols:
        # Polynomial only on continuous columns, degree=2
        poly_transform = Pipeline(steps=[
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("cont_poly", poly_transform, cont_cols))
        remainder = "passthrough"
    else:
        # Just scale continuous columns, pass others
        transformers.append(("scale", StandardScaler(), cont_cols))
        remainder = "passthrough"

    preproc = ColumnTransformer(transformers=transformers, remainder=remainder)

    # Model
    lr = LogisticRegression(max_iter=max_iter, class_weight="balanced",
                            solver="lbfgs", n_jobs=-1)

    pipe = Pipeline(steps=[
        ("pre", preproc),
        ("lr",  lr),
    ])

    # Hyperparameter grid (tune C)
    param_grid = {
        "lr__C": [0.02, 0.05, 0.1, 0.5, 1, 2, 5, 10]
    }

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    print(f"[lr] grid search... poly={poly}, cv={cv}")
    gs.fit(X_tr, y_tr)

    best = gs.best_estimator_
    print("[lr] best params:", gs.best_params_)

    # Evaluate on test
    y_pred = best.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="macro")
    print(f"\n[lr] TEST accuracy: {acc:.4f}  macro-F1: {f1:.4f}")
    print(classification_report(le.inverse_transform(y_te),
                                le.inverse_transform(y_pred)))

    # Save artifacts (compatible with your serving pipeline name)
    joblib.dump(best, MODELS / "severity_lr.joblib")
    X.columns.to_series().to_csv(MODELS / "feature_order.csv", index=False)
    with open(MODELS / "label_classes.json", "w") as f:
        json.dump(le.classes_.tolist(), f)

    with open(PROC / "training_metrics_lr.json", "w") as f:
        json.dump(
            {"acc": float(acc), "macro_f1": float(f1),
             "best_params": gs.best_params_, "poly": poly,
             "cont_cols_count": len(cont_cols)},
            f, indent=2
        )

    print("\n[lr] Saved model: severity_lr.joblib")
    print("     Saved feature_order.csv, label_classes.json, training_metrics_lr.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Boosted Logistic Regression training (scaling + tuning + optional poly).")
    ap.add_argument("--poly", action="store_true",
                    help="Enable PolynomialFeatures(degree=2) on continuous columns.")
    ap.add_argument("--cv", type=int, default=5, help="CV folds for GridSearchCV (default 5).")
    ap.add_argument("--seed", type=int, default=42, help="Random state.")
    ap.add_argument("--max-iter", type=int, default=5000, help="LR max_iter (default 5000).")
    args = ap.parse_args()
    main(poly=args.poly, cv=args.cv, random_state=args.seed, max_iter=args.max_iter)
