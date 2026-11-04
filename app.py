"""
Streamlit app for Assignment 2 — Breast Cancer classification demo.

Upload the provided test_data.csv (or any CSV with the same 30 feature
columns + a 'target' column), pick a trained model, and view evaluation
metrics, the confusion matrix, and the classification report without any
retraining happening in the app.
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = "model"
TARGET_COL = "target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Breast Cancer Classifier Demo", layout="wide")


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    models = {name: joblib.load(os.path.join(MODEL_DIR, fname)) for name, fname in MODEL_FILES.items()}
    with open(os.path.join(MODEL_DIR, "target_names.json")) as f:
        target_names = json.load(f)
    return scaler, feature_names, models, target_names


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    st.title("🩺 Breast Cancer Classification — Model Demo")
    st.caption(
        "BITS Pilani WILP M.Tech (AIML) — Machine Learning Assignment 2. "
        "Dataset: Breast Cancer Wisconsin (Diagnostic), UCI ML Repository."
    )

    scaler, feature_names, models, target_names = load_artifacts()

    st.sidebar.header("1. Upload test data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV with the 30 feature columns + a 'target' column", type=["csv"]
    )

    st.sidebar.header("2. Choose a model")
    model_name = st.sidebar.selectbox("Model", list(models.keys()))

    show_all = st.sidebar.checkbox("Also compare all 5 models on this data", value=False)

    if uploaded_file is None:
        st.info("⬅️ Upload a test CSV from the sidebar to get started. You can use the provided `test_data.csv`.")
        st.subheader("Expected columns")
        st.write(feature_names + [TARGET_COL])
        return

    df = pd.read_csv(uploaded_file)

    missing_cols = [c for c in feature_names + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required column(s): {missing_cols}")
        return

    features = df[feature_names]
    y_true = df[TARGET_COL]
    scaled_features = scaler.transform(features)

    st.subheader(f"📊 Results — {model_name}")
    model = models[model_name]
    y_pred = model.predict(scaled_features)
    y_proba = model.predict_proba(scaled_features)[:, 1]

    metrics = compute_metrics(y_true, y_pred, y_proba)

    cols = st.columns(6)
    for col, (metric_name, value) in zip(cols, metrics.items()):
        col.metric(metric_name, f"{value:.4f}")

    left, right = st.columns(2)

    with left:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.markdown("**Classification Report**")
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(4))

    if show_all:
        st.subheader("📈 Comparison across all 5 models (this uploaded data)")
        rows = []
        for name, mdl in models.items():
            pred = mdl.predict(scaled_features)
            proba = mdl.predict_proba(scaled_features)[:, 1]
            m = compute_metrics(y_true, pred, proba)
            m["ML Model Name"] = name
            rows.append(m)
        comparison_df = pd.DataFrame(rows).set_index("ML Model Name")
        comparison_df = comparison_df[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]].round(4)
        st.dataframe(comparison_df)
        st.bar_chart(comparison_df)


if __name__ == "__main__":
    main()
