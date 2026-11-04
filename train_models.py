"""
train_models.py
----------------
Loads the Breast Cancer Wisconsin (Diagnostic) dataset (UCI / sklearn),
trains 5 classification models, evaluates them, and persists:
  - trained models (model/*.joblib)
  - the fitted StandardScaler (model/scaler.joblib)
  - the feature column order (model/feature_names.joblib)
  - a held-out test split as test_data.csv (unscaled, with target column)
  - a metrics comparison table as model_metrics.csv

Run this once to (re)generate all artifacts used by app.py:
    python train_models.py
"""

import json
import pandas as pd
from joblib import dump
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

import os

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COL = "target"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df = df.rename(columns={"target": TARGET_COL})

feature_names = [c for c in df.columns if c != TARGET_COL]

print(f"Dataset shape: {df.shape}")
print(f"Number of features: {len(feature_names)}")
print(f"Class distribution:\n{df[TARGET_COL].value_counts()}")

X = df[feature_names]
y = df[TARGET_COL]

# ---------------------------------------------------------------------------
# 2. Train / test split (test split is saved as the CSV used by the Streamlit app)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

test_df = X_test.copy()
test_df[TARGET_COL] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print(f"Saved test_data.csv with shape {test_df.shape}")

# ---------------------------------------------------------------------------
# 3. Scale features (fit scaler on train only)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
dump(feature_names, os.path.join(MODEL_DIR, "feature_names.joblib"))

# ---------------------------------------------------------------------------
# 4. Define models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

file_slugs = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}

# ---------------------------------------------------------------------------
# 5. Train, evaluate, and save each model
# ---------------------------------------------------------------------------
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(f"{name}: {metrics}")

    dump(model, os.path.join(MODEL_DIR, f"{file_slugs[name]}.joblib"))

# ---------------------------------------------------------------------------
# 6. Save metrics comparison table (used to populate the README)
# ---------------------------------------------------------------------------
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("model_metrics.csv", index=False)
print("\nComparison table:\n", metrics_df.to_string(index=False))

with open(os.path.join(MODEL_DIR, "target_names.json"), "w") as f:
    json.dump(list(data.target_names), f)

print("\nAll models and artifacts saved under 'model/'.")
