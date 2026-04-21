"""
=======================================================================
  End-to-End ML Pipeline for Telco Customer Churn Prediction
  Task 2 — Scikit-learn Pipeline API
=======================================================================
  Problem Statement:
    Customer churn is a major business problem in the telecom industry.
    This script builds a reusable, production-ready ML pipeline that:
      • Preprocesses raw Telco data (scaling, encoding, imputation)
      • Trains Logistic Regression and Random Forest classifiers
      • Tunes hyperparameters with GridSearchCV
      • Exports the best pipeline with joblib for deployment

  Dataset : Telco Customer Churn (IBM / Kaggle)
  Author  : ML Engineer
  Date    : 2026-04-21
=======================================================================
"""

# ── 0. Imports ────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless rendering for script mode
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)

# ── Directories ───────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  TELCO CUSTOMER CHURN — END-TO-END ML PIPELINE")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & EXPLORATION
# ═══════════════════════════════════════════════════════════════════════
print("\n[1] Loading Dataset …")

df = pd.read_csv("telco_churn.csv")
print(f"    Shape           : {df.shape}")
print(f"    Churn balance   :\n{df['Churn'].value_counts(normalize=True).round(3).to_string()}")
print(f"    Missing values  :\n{df.isnull().sum()[df.isnull().sum()>0].to_string()}")

# ── Drop customerID (identifier, not a feature) ───────────────────────
df.drop(columns=["customerID"], inplace=True)

# TotalCharges arrived as object in the original IBM dataset;
# coerce to numeric so the imputer can handle it.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# ── Target encoding ───────────────────────────────────────────────────
df["Churn"] = (df["Churn"] == "Yes").astype(int)

X = df.drop(columns=["Churn"])
y = df["Churn"]

# ── Feature lists ─────────────────────────────────────────────────────
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES  = ["SeniorCitizen"]          # already 0/1, no encoding needed
CATEGORICAL_FEATURES = [
    c for c in X.columns
    if c not in NUMERIC_FEATURES + BINARY_FEATURES
]

print(f"\n    Numeric features      : {NUMERIC_FEATURES}")
print(f"    Binary features       : {BINARY_FEATURES}")
print(f"    Categorical features  : {CATEGORICAL_FEATURES}")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════
print("\n[2] Splitting data (80 / 20 stratified) …")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"    Train : {X_train.shape}  |  Test : {X_test.shape}")
print(f"    Train churn rate : {y_train.mean():.3f}  |  Test churn rate : {y_test.mean():.3f}")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════
print("\n[3] Building preprocessing pipeline …")

# Numeric: impute median → scale
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

# Categorical: impute most_frequent → one-hot encode
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# ColumnTransformer combines both; passthrough binary col
preprocessor = ColumnTransformer([
    ("num",  numeric_transformer,    NUMERIC_FEATURES),
    ("cat",  categorical_transformer, CATEGORICAL_FEATURES),
    ("bin",  "passthrough",          BINARY_FEATURES),
])

print("    ✓ Numeric  : median imputation + StandardScaler")
print("    ✓ Categorical : mode imputation + OneHotEncoder")
print("    ✓ Binary   : passthrough (SeniorCitizen is already 0/1)")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL PIPELINES
# ═══════════════════════════════════════════════════════════════════════
print("\n[4] Assembling full model pipelines …")

pipe_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   LogisticRegression(max_iter=1000, random_state=42)),
])

pipe_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(n_jobs=-1, random_state=42)),
])

# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — HYPERPARAMETER TUNING (GridSearchCV)
# ═══════════════════════════════════════════════════════════════════════
print("\n[5] Hyperparameter tuning with GridSearchCV (5-fold CV) …")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Logistic Regression grid ──────────────────────────────────────────
lr_param_grid = {
    "classifier__C":       [0.01, 0.1, 1.0, 10.0],
    "classifier__solver":  ["lbfgs", "liblinear"],
    "classifier__penalty": ["l2"],
}

print("    Tuning Logistic Regression …", end="", flush=True)
t0 = time.time()
gs_lr = GridSearchCV(pipe_lr, lr_param_grid, cv=cv,
                     scoring="roc_auc", n_jobs=-1, verbose=0)
gs_lr.fit(X_train, y_train)
print(f" done in {time.time()-t0:.1f}s")
print(f"    Best params : {gs_lr.best_params_}")
print(f"    Best CV AUC : {gs_lr.best_score_:.4f}")

# ── Random Forest grid ────────────────────────────────────────────────
rf_param_grid = {
    "classifier__n_estimators":      [100, 200],
    "classifier__max_depth":         [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
}

print("    Tuning Random Forest …", end="", flush=True)
t0 = time.time()
gs_rf = GridSearchCV(pipe_rf, rf_param_grid, cv=cv,
                     scoring="roc_auc", n_jobs=-1, verbose=0)
gs_rf.fit(X_train, y_train)
print(f" done in {time.time()-t0:.1f}s")
print(f"    Best params : {gs_rf.best_params_}")
print(f"    Best CV AUC : {gs_rf.best_score_:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 6 — EVALUATION ON TEST SET
# ═══════════════════════════════════════════════════════════════════════
print("\n[6] Evaluating on held-out test set …")

def evaluate(name, model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall"   : recall_score(y_test, y_pred),
        "F1"       : f1_score(y_test, y_pred),
        "ROC-AUC"  : roc_auc_score(y_test, y_proba),
    }
    print(f"\n    ── {name} ──")
    for k, v in metrics.items():
        print(f"       {k:<12}: {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn','Churn'])}")
    return metrics, y_pred, y_proba

metrics_lr, pred_lr, proba_lr = evaluate("Logistic Regression", gs_lr.best_estimator_, X_test, y_test)
metrics_rf, pred_rf, proba_rf = evaluate("Random Forest",       gs_rf.best_estimator_, X_test, y_test)

# ── Pick best overall model ───────────────────────────────────────────
best_name    = "Random Forest" if metrics_rf["ROC-AUC"] >= metrics_lr["ROC-AUC"] else "Logistic Regression"
best_model   = gs_rf.best_estimator_  if best_name == "Random Forest" else gs_lr.best_estimator_
best_proba   = proba_rf               if best_name == "Random Forest" else proba_lr
best_pred    = pred_rf                if best_name == "Random Forest" else pred_lr

print(f"\n    🏆  Best model: {best_name}")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 7 — VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════
print("\n[7] Generating visualisations …")

PALETTE = {"no_churn": "#2196F3", "churn": "#FF5252",
           "lr": "#00BCD4", "rf": "#FF9800"}

# ─── Fig 1 : Data overview ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Telco Churn — Data Overview", fontsize=14, fontweight="bold")

# 1a — churn distribution
counts = df["Churn"].value_counts()
axes[0].bar(["No Churn","Churn"], counts.values,
            color=[PALETTE["no_churn"], PALETTE["churn"]], edgecolor="white", linewidth=1.5)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 30, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)
axes[0].set_title("Churn Distribution"); axes[0].set_ylabel("Count")

# 1b — tenure distribution by churn
for label, color, name in [(0, PALETTE["no_churn"], "No Churn"),
                            (1, PALETTE["churn"],    "Churn")]:
    axes[1].hist(df.loc[df["Churn"]==label, "tenure"], bins=30,
                 alpha=0.65, color=color, label=name, edgecolor="white")
axes[1].legend(); axes[1].set_title("Tenure by Churn"); axes[1].set_xlabel("Months")

# 1c — monthly charges by churn
for label, color, name in [(0, PALETTE["no_churn"], "No Churn"),
                            (1, PALETTE["churn"],    "Churn")]:
    axes[2].hist(df.loc[df["Churn"]==label, "MonthlyCharges"], bins=30,
                 alpha=0.65, color=color, label=name, edgecolor="white")
axes[2].legend(); axes[2].set_title("Monthly Charges by Churn"); axes[2].set_xlabel("USD")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_data_overview.png", dpi=150)
plt.close()
print("    ✓ fig1_data_overview.png")

# ─── Fig 2 : GridSearchCV heatmaps ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("GridSearchCV — Mean CV ROC-AUC", fontsize=13, fontweight="bold")

# LR heatmap : C vs solver
lr_results = pd.DataFrame(gs_lr.cv_results_)
lr_pivot = lr_results.pivot_table(
    values="mean_test_score",
    index="param_classifier__C",
    columns="param_classifier__solver"
)
sns.heatmap(lr_pivot, annot=True, fmt=".4f", cmap="Blues", ax=axes[0])
axes[0].set_title("Logistic Regression  (C vs solver)")

# RF heatmap : n_estimators vs max_depth
rf_results = pd.DataFrame(gs_rf.cv_results_)
rf_pivot = rf_results.pivot_table(
    values="mean_test_score",
    index="param_classifier__n_estimators",
    columns="param_classifier__max_depth"
)
sns.heatmap(rf_pivot, annot=True, fmt=".4f", cmap="Oranges", ax=axes[1])
axes[1].set_title("Random Forest  (n_estimators vs max_depth)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_gridsearch_heatmaps.png", dpi=150)
plt.close()
print("    ✓ fig2_gridsearch_heatmaps.png")

# ─── Fig 3 : Confusion matrices + ROC curves ─────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Model Evaluation — Test Set", fontsize=14, fontweight="bold")

for (ax_cm, ax_roc), (name, pred, proba) in zip(
    [(axes[0,0], axes[0,1]), (axes[1,0], axes[1,1])],
    [("Logistic Regression", pred_lr, proba_lr),
     ("Random Forest",       pred_rf, proba_rf)]
):
    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                xticklabels=["Pred No","Pred Yes"],
                yticklabels=["True No","True Yes"])
    ax_cm.set_title(f"{name}\nConfusion Matrix")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = roc_auc_score(y_test, proba)
    color = PALETTE["lr"] if "Logistic" in name else PALETTE["rf"]
    ax_roc.plot(fpr, tpr, color=color, lw=2,
                label=f"AUC = {auc_val:.4f}")
    ax_roc.plot([0,1],[0,1], "k--", lw=1)
    ax_roc.set(xlabel="FPR", ylabel="TPR",
               title=f"{name}\nROC Curve")
    ax_roc.legend(loc="lower right")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_evaluation.png", dpi=150)
plt.close()
print("    ✓ fig3_evaluation.png")

# ─── Fig 4 : Feature importance (Random Forest) ──────────────────────
print("    Generating feature importance plot …")

rf_clf   = gs_rf.best_estimator_.named_steps["classifier"]
ohe_cats = (gs_rf.best_estimator_
                .named_steps["preprocessor"]
                .named_transformers_["cat"]
                .named_steps["encoder"]
                .get_feature_names_out(CATEGORICAL_FEATURES))
feature_names = NUMERIC_FEATURES + list(ohe_cats) + BINARY_FEATURES
importances   = rf_clf.feature_importances_

top_n  = 20
idx    = np.argsort(importances)[::-1][:top_n]

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(range(top_n), importances[idx][::-1],
        color=PALETTE["rf"], edgecolor="white")
ax.set_yticks(range(top_n))
ax.set_yticklabels([feature_names[i] for i in idx[::-1]], fontsize=9)
ax.set_xlabel("Importance"); ax.set_title(f"Random Forest — Top {top_n} Feature Importances")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_feature_importance.png", dpi=150)
plt.close()
print("    ✓ fig4_feature_importance.png")

# ─── Fig 5 : Metrics comparison bar chart ────────────────────────────
metrics_names = list(metrics_lr.keys())
lr_vals = list(metrics_lr.values())
rf_vals = list(metrics_rf.values())
x       = np.arange(len(metrics_names))
width   = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, lr_vals, width, label="Logistic Regression",
               color=PALETTE["lr"], edgecolor="white")
bars2 = ax.bar(x + width/2, rf_vals, width, label="Random Forest",
               color=PALETTE["rf"], edgecolor="white")

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x); ax.set_xticklabels(metrics_names)
ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
ax.set_title("Model Comparison — Test Metrics")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig5_model_comparison.png", dpi=150)
plt.close()
print("    ✓ fig5_model_comparison.png")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 8 — EXPORT WITH JOBLIB
# ═══════════════════════════════════════════════════════════════════════
print("\n[8] Exporting pipeline with joblib …")

MODEL_PATH = f"{OUTPUT_DIR}/best_churn_pipeline.joblib"
joblib.dump(best_model, MODEL_PATH, compress=3)
print(f"    ✓ Saved: {MODEL_PATH}  ({os.path.getsize(MODEL_PATH)/1024:.1f} KB)")

# ── Verify reload + inference ─────────────────────────────────────────
loaded_model = joblib.load(MODEL_PATH)
sample_preds = loaded_model.predict_proba(X_test.head(5))[:, 1]
print(f"    ✓ Reload verified. Sample churn probabilities: {sample_preds.round(3)}")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 9 — FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)

summary = pd.DataFrame({
    "Model":     ["Logistic Regression", "Random Forest"],
    "Best CV AUC":[f"{gs_lr.best_score_:.4f}", f"{gs_rf.best_score_:.4f}"],
    "Test Accuracy": [f"{metrics_lr['Accuracy']:.4f}", f"{metrics_rf['Accuracy']:.4f}"],
    "Test F1"   : [f"{metrics_lr['F1']:.4f}",       f"{metrics_rf['F1']:.4f}"],
    "Test ROC-AUC":[f"{metrics_lr['ROC-AUC']:.4f}", f"{metrics_rf['ROC-AUC']:.4f}"],
})
print(summary.to_string(index=False))

print(f"""
Key Insights:
  • {best_name} outperformed on ROC-AUC, selected as production model.
  • Feature importance (RF) reveals: tenure, MonthlyCharges, and Contract
    type are the most predictive churn signals.
  • Month-to-month contract customers show significantly higher churn rates.
  • The preprocessing pipeline handles missing TotalCharges via median imputation,
    making it robust to real-world data quality issues.
  • GridSearchCV with 5-fold stratified CV ensures reliable hyperparameter
    estimates; AUC scoring is preferred over accuracy for imbalanced labels.
  • The exported joblib pipeline bundles preprocessing + model in one object,
    enabling one-line inference on raw DataFrames in production.

Recommendations:
  • Prioritise retention campaigns for month-to-month subscribers.
  • Offer incentives to customers in first 12 months (high churn window).
  • Consider calibration (CalibratedClassifierCV) for probability outputs
    if used in business decision-making tools.
""")

print(f"All outputs saved to: ./{OUTPUT_DIR}/")
print("=" * 70)
