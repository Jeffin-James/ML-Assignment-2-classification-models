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
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
SAMPLE_CSV_PATH = "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

MODEL_DESCRIPTIONS = {
    "Logistic Regression": "Linear model estimating class probability via the logistic function. Fast, interpretable.",
    "Decision Tree": "Rule-based tree splitting on feature thresholds. Easy to visualize and explain.",
    "KNN": "Classifies a sample by majority vote among its K nearest neighbours in feature space.",
    "Naive Bayes": "Probabilistic classifier assuming feature independence given the class (Gaussian likelihood).",
    "Support Vector Machine": "Finds the optimal separating boundary (RBF kernel) that maximizes the margin between classes.",
}

BRAND_COLORWAY = ["#1e4d8c", "#2e9e5b", "#d97706", "#c0392b", "#6b4fbb"]
ACCENT_BLUE = "#1e4d8c"

st.set_page_config(
    page_title="Breast Cancer Classifier Demo",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1150px;}

        .app-header {
            background: linear-gradient(90deg, #0f2847 0%, #1e4d8c 100%);
            padding: 22px 28px;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            color: #ffffff;
        }
        .app-header .bits-id {
            font-size: 0.8rem;
            letter-spacing: 0.05em;
            color: #a9c3e8;
            text-transform: uppercase;
            margin: 0;
        }
        .app-header h1 {
            color: #ffffff;
            font-size: 1.6rem;
            font-weight: 700;
            margin: 4px 0 2px 0;
        }
        .app-header p {
            color: #cfe0f5;
            font-size: 0.9rem;
            margin: 0;
        }

        div[data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 10px;
            padding: 14px 8px;
            text-align: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        div[data-testid="stMetric"] label {color: var(--text-color) !important;}
        div[data-testid="stMetricValue"] div {color: var(--text-color) !important;}
        div[data-testid="stMetricLabel"] {justify-content: center;}
        div[data-testid="stMetricValue"] {justify-content: center;}

        section[data-testid="stSidebar"] {
            background-color: var(--secondary-background-color);
            border-right: 1px solid rgba(128, 128, 128, 0.25);
        }
        section[data-testid="stSidebar"] * {color: var(--text-color);}

        .stTabs [data-baseweb="tab-list"] {gap: 4px;}
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 18px;
            font-weight: 500;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def render_single_model_results(model_name, model, scaled_features, y_true, target_names):
    y_pred = model.predict(scaled_features)
    y_proba = model.predict_proba(scaled_features)[:, 1]
    metrics = compute_metrics(y_true, y_pred, y_proba)

    st.markdown(f"#### Results — {model_name}")
    st.caption(MODEL_DESCRIPTIONS.get(model_name, ""))

    cols = st.columns(6)
    for col, (metric_name, value) in zip(cols, metrics.items()):
        col.metric(metric_name, f"{value:.4f}")

    st.write("")
    left, right = st.columns(2)

    with left:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=[f"Pred: {t}" for t in target_names],
            y=[f"Actual: {t}" for t in target_names],
        )
        fig_cm.update_traces(textfont_size=18)
        fig_cm.update_layout(
            height=380,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with right:
        st.markdown("**Metric Profile**")
        categories = list(metrics.keys())
        values = list(metrics.values())
        fig_radar = go.Figure(
            data=go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                line_color=ACCENT_BLUE,
                fillcolor="rgba(30, 77, 140, 0.25)",
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=380,
            margin=dict(l=30, r=30, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("**Classification Report**")
    report_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=target_names, output_dict=True)).transpose().round(4)
    styled_report = report_df.style.background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"])
    st.dataframe(styled_report, use_container_width=True)


def render_comparison(models, scaled_features, y_true):
    st.markdown("#### Comparison across all 5 models")
    rows = []
    for name, mdl in models.items():
        pred = mdl.predict(scaled_features)
        proba = mdl.predict_proba(scaled_features)[:, 1]
        m = compute_metrics(y_true, pred, proba)
        m["ML Model Name"] = name
        rows.append(m)
    comparison_df = pd.DataFrame(rows).set_index("ML Model Name")
    comparison_df = comparison_df[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]].round(4)

    best_model = comparison_df["Accuracy"].idxmax()
    st.success(f"🏆 Best Accuracy on this data: **{best_model}** ({comparison_df.loc[best_model, 'Accuracy']:.4f})")

    tab_bar, tab_radar, tab_table = st.tabs(["Bar Chart", "Radar Chart", "Table"])

    with tab_bar:
        melted = comparison_df.reset_index().melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
        fig_bar = px.bar(
            melted,
            x="Metric",
            y="Score",
            color="ML Model Name",
            barmode="group",
            color_discrete_sequence=BRAND_COLORWAY,
            text_auto=".3f",
        )
        fig_bar.update_layout(
            height=440,
            yaxis_range=[0, 1.05],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_radar:
        categories = list(comparison_df.columns)
        fig_radar = go.Figure()
        for i, (name, row) in enumerate(comparison_df.iterrows()):
            values = row.tolist()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=name,
                    line_color=BRAND_COLORWAY[i % len(BRAND_COLORWAY)],
                    opacity=0.8,
                )
            )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(l=30, r=30, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab_table:
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, color="#c6f6d5").background_gradient(cmap="Blues"),
            use_container_width=True,
        )


def main():
    st.markdown(
        """
        <div class="app-header">
            <p class="bits-id">BITS ID: 2026AJ05809</p>
            <h1>🩺 Breast Cancer Classification</h1>
            <p>Machine Learning Assignment 2 — Model Evaluation Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scaler, feature_names, models, target_names = load_artifacts()

    with st.sidebar:
        st.markdown("### Controls")
        if os.path.exists(SAMPLE_CSV_PATH):
            with open(SAMPLE_CSV_PATH, "rb") as f:
                st.download_button(
                    "⬇️ Download sample CSV",
                    data=f,
                    file_name="test_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        uploaded_file = st.file_uploader("Upload test data", type=["csv"])
        model_name = st.selectbox("Model", list(models.keys()))
        st.caption(MODEL_DESCRIPTIONS.get(model_name, ""))

    if uploaded_file is None:
        st.info("⬅️ Upload a CSV from the sidebar to view results, or download the sample file first.")
        return

    df = pd.read_csv(uploaded_file)

    missing_cols = [c for c in feature_names + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required column(s): {missing_cols}")
        return

    features = df[feature_names]
    y_true = df[TARGET_COL]
    scaled_features = scaler.transform(features)

    tab_single, tab_compare = st.tabs(["🔍 Single Model Evaluation", "📊 Compare All Models"])
    with tab_single:
        render_single_model_results(model_name, models[model_name], scaled_features, y_true, target_names)
    with tab_compare:
        render_comparison(models, scaled_features, y_true)


if __name__ == "__main__":
    main()
