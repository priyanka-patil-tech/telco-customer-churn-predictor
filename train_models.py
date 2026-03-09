"""
MSIS 522 HW1 - Training Script
Telco Customer Churn - Binary Classification
Trains: Logistic Regression, Decision Tree, Random Forest, XGBoost, MLP (sklearn)
Saves all models, metrics, ROC data, SHAP values, and MLP history.
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

df = df.drop("customerID", axis=1)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

X_raw = df.drop("Churn", axis=1)
y = df["Churn"]

numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
binary_cols = ["SeniorCitizen"]
categorical_cols = [c for c in X_raw.columns if c not in numerical_cols + binary_cols]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("bin", "passthrough", binary_cols),
    ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols),
])

X_processed = preprocessor.fit_transform(X_raw)
ohe = preprocessor.named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
feature_names = numerical_cols + binary_cols + cat_names

X_df = pd.DataFrame(X_processed, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
print(f"  Class balance (train) - Churn: {y_train.mean():.3f}")

# Save preprocessing artifacts
joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
joblib.dump(X_raw, MODELS_DIR / "X_raw.joblib")
np.savez(MODELS_DIR / "test_data.npz",
         X_test=X_test.values, y_test=y_test.values)
with open(MODELS_DIR / "feature_names.json", "w") as f:
    json.dump(feature_names, f)
# Save raw df for EDA in app
df.to_csv(MODELS_DIR / "processed_df.csv", index=False)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def compute_metrics(model, Xt, yt, name):
    yp = model.predict(Xt)
    yprob = model.predict_proba(Xt)[:, 1]
    m = {
        "accuracy":  round(float(accuracy_score(yt, yp)), 4),
        "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
        "recall":    round(float(recall_score(yt, yp, zero_division=0)), 4),
        "f1":        round(float(f1_score(yt, yp, zero_division=0)), 4),
        "auc_roc":   round(float(roc_auc_score(yt, yprob)), 4),
    }
    print(f"  {name}: Acc={m['accuracy']}, F1={m['f1']}, AUC={m['auc_roc']}")
    return m


def get_roc(model, Xt, yt):
    yprob = model.predict_proba(Xt)[:, 1]
    fpr, tpr, _ = roc_curve(yt, yprob)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


all_metrics = {}
roc_data = {}
best_params = {}


# ─────────────────────────────────────────────
# 2.2 LOGISTIC REGRESSION BASELINE
# ─────────────────────────────────────────────
print("\n[2.2] Logistic Regression...")
lr = LogisticRegression(
    random_state=RANDOM_STATE, max_iter=1000,
    class_weight="balanced", solver="lbfgs", C=1.0
)
lr.fit(X_train, y_train)
joblib.dump(lr, MODELS_DIR / "logistic_regression.joblib")
all_metrics["Logistic Regression"] = compute_metrics(lr, X_test, y_test, "LR")
roc_data["Logistic Regression"] = get_roc(lr, X_test, y_test)
best_params["Logistic Regression"] = {"C": 1.0, "solver": "lbfgs", "class_weight": "balanced"}


# ─────────────────────────────────────────────
# 2.3 DECISION TREE (5-fold CV GridSearch)
# ─────────────────────────────────────────────
print("\n[2.3] Decision Tree with GridSearchCV...")
dt_grid = {
    "max_depth": [3, 5, 7, 10],
    "min_samples_leaf": [5, 10, 20, 50],
}
dt_gs = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
    dt_grid, cv=5, scoring="f1", n_jobs=-1
)
dt_gs.fit(X_train, y_train)
best_dt = dt_gs.best_estimator_
print(f"  Best params: {dt_gs.best_params_}")
joblib.dump(best_dt, MODELS_DIR / "decision_tree.joblib")
all_metrics["Decision Tree"] = compute_metrics(best_dt, X_test, y_test, "DT")
roc_data["Decision Tree"] = get_roc(best_dt, X_test, y_test)
best_params["Decision Tree"] = dt_gs.best_params_


# ─────────────────────────────────────────────
# 2.4 RANDOM FOREST (5-fold CV GridSearch)
# ─────────────────────────────────────────────
print("\n[2.4] Random Forest with GridSearchCV...")
rf_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 8],
}
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
    rf_grid, cv=5, scoring="f1", n_jobs=1
)
rf_gs.fit(X_train, y_train)
best_rf = rf_gs.best_estimator_
print(f"  Best params: {rf_gs.best_params_}")
joblib.dump(best_rf, MODELS_DIR / "random_forest.joblib")
all_metrics["Random Forest"] = compute_metrics(best_rf, X_test, y_test, "RF")
roc_data["Random Forest"] = get_roc(best_rf, X_test, y_test)
best_params["Random Forest"] = rf_gs.best_params_


# ─────────────────────────────────────────────
# 2.5 XGBOOST (5-fold CV GridSearch)
# ─────────────────────────────────────────────
print("\n[2.5] XGBoost with GridSearchCV...")
scale_pw = float(sum(y_train == 0) / sum(y_train == 1))
xgb_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1],
}
xgb_gs = GridSearchCV(
    xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pw,
        eval_metric="logloss",
        verbosity=0,
    ),
    xgb_grid, cv=5, scoring="f1", n_jobs=-1
)
xgb_gs.fit(X_train, y_train)
best_xgb = xgb_gs.best_estimator_
print(f"  Best params: {xgb_gs.best_params_}")
joblib.dump(best_xgb, MODELS_DIR / "xgboost_model.joblib")
all_metrics["XGBoost"] = compute_metrics(best_xgb, X_test, y_test, "XGB")
roc_data["XGBoost"] = get_roc(best_xgb, X_test, y_test)
best_params["XGBoost"] = xgb_gs.best_params_


# ─────────────────────────────────────────────
# 2.6 MLP NEURAL NETWORK (sklearn)
# ─────────────────────────────────────────────
print("\n[2.6] MLP Neural Network...")
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 128),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=300,
    random_state=RANDOM_STATE,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    verbose=False,
)
mlp.fit(X_train, y_train)
joblib.dump(mlp, MODELS_DIR / "mlp_model.joblib")
all_metrics["MLP Neural Network"] = compute_metrics(mlp, X_test, y_test, "MLP")
roc_data["MLP Neural Network"] = get_roc(mlp, X_test, y_test)
best_params["MLP Neural Network"] = {
    "hidden_layer_sizes": "(128, 128)",
    "activation": "relu",
    "solver": "adam",
    "learning_rate_init": 0.001,
}

# MLP training history
mlp_history = {
    "loss_curve": [round(v, 6) for v in mlp.loss_curve_],
    "val_scores":  [round(v, 6) for v in mlp.validation_scores_],
    "n_iter": mlp.n_iter_,
}
with open(MODELS_DIR / "mlp_history.json", "w") as f:
    json.dump(mlp_history, f)
print(f"  MLP converged in {mlp.n_iter_} iterations")


# ─────────────────────────────────────────────
# BONUS: MLP HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
print("\n[BONUS] MLP Hyperparameter Tuning (GridSearchCV)...")
mlp_hp_grid = {
    "hidden_layer_sizes": [(64, 64), (128, 128), (64, 128, 64), (256, 128)],
    "learning_rate_init": [0.001, 0.005, 0.01],
    "alpha": [0.0001, 0.001],  # L2 regularization
}
mlp_hp_gs = GridSearchCV(
    MLPClassifier(
        activation="relu", solver="adam", max_iter=150,
        random_state=RANDOM_STATE, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=10,
    ),
    mlp_hp_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1,
)
mlp_hp_gs.fit(X_train, y_train)
print(f"  Best MLP params: {mlp_hp_gs.best_params_}, F1={mlp_hp_gs.best_score_:.4f}")

best_mlp_tuned = mlp_hp_gs.best_estimator_
joblib.dump(best_mlp_tuned, MODELS_DIR / "mlp_best_tuned.joblib")

# Save detailed HP results for visualization
cv_results = mlp_hp_gs.cv_results_
mlp_hp_results = {
    "best_params": str(mlp_hp_gs.best_params_),
    "best_f1_cv": round(float(mlp_hp_gs.best_score_), 4),
    "cv_results": {
        "params": [str(p) for p in cv_results["params"]],
        "mean_test_score": [round(float(v), 4) for v in cv_results["mean_test_score"]],
        "std_test_score":  [round(float(v), 4) for v in cv_results["std_test_score"]],
    },
}
with open(MODELS_DIR / "mlp_hp_results.json", "w") as f:
    json.dump(mlp_hp_results, f, indent=2)


# ─────────────────────────────────────────────
# 3. SHAP ANALYSIS (using XGBoost)
# ─────────────────────────────────────────────
print("\n[3] SHAP Analysis...")
rng = np.random.RandomState(RANDOM_STATE)
shap_idx = rng.choice(len(X_test), min(500, len(X_test)), replace=False)
X_shap = X_test.iloc[shap_idx]

explainer = shap.TreeExplainer(best_xgb)
shap_vals = explainer.shap_values(X_shap)
exp_val = float(np.asarray(explainer.expected_value).ravel()[0])

np.savez(
    MODELS_DIR / "shap_data.npz",
    shap_values=shap_vals,
    X_sample=X_shap.values,
    expected_value=np.array([exp_val]),
)
with open(MODELS_DIR / "shap_feature_names.json", "w") as f:
    json.dump(feature_names, f)
print(f"  SHAP computed on {len(X_shap)} samples")


# ─────────────────────────────────────────────
# SAVE ALL METRICS & ROC DATA
# ─────────────────────────────────────────────
output = {
    "metrics":      all_metrics,
    "roc_data":     roc_data,
    "best_params":  {k: {str(kk): str(vv) for kk, vv in v.items()} for k, v in best_params.items()},
    "feature_names": feature_names,
    "class_balance": {
        "train_churn_rate": round(float(y_train.mean()), 4),
        "test_churn_rate":  round(float(y_test.mean()), 4),
        "n_train": int(len(y_train)),
        "n_test":  int(len(y_test)),
    },
}
with open(MODELS_DIR / "metrics.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n" + "=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)
print(f"Models saved to: {MODELS_DIR}")
print("\nModel Comparison (Test Set):")
print(f"{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("-" * 58)
for name, m in all_metrics.items():
    print(f"{name:<25} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
          f"{m['recall']:>6.4f} {m['f1']:>6.4f} {m['auc_roc']:>6.4f}")
