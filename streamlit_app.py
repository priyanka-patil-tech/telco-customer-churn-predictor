"""
MSIS 522 HW1 - Telco Customer Churn Prediction
Streamlit App: 4-tab layout covering the complete data science workflow.
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"
DATA_PATH  = Path(__file__).parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RANDOM_STATE = 42

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📡",
    layout="wide",
)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree":        "decision_tree.joblib",
    "Random Forest":        "random_forest.joblib",
    "XGBoost":              "xgboost_model.joblib",
    "MLP Neural Network":   "mlp_model.joblib",
}


# ─────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        name: joblib.load(MODELS_DIR / fname)
        for name, fname in MODEL_FILES.items()
        if (MODELS_DIR / fname).exists()
    }

@st.cache_resource
def load_preprocessor():
    return joblib.load(MODELS_DIR / "preprocessor.joblib")

@st.cache_data
def load_metrics():
    with open(MODELS_DIR / "metrics.json") as f:
        return json.load(f)

@st.cache_data
def load_mlp_history():
    with open(MODELS_DIR / "mlp_history.json") as f:
        return json.load(f)

@st.cache_data
def load_mlp_hp():
    with open(MODELS_DIR / "mlp_hp_results.json") as f:
        return json.load(f)

@st.cache_data
def load_shap_data():
    data = np.load(MODELS_DIR / "shap_data.npz", allow_pickle=True)
    with open(MODELS_DIR / "shap_feature_names.json") as f:
        feat_names = json.load(f)
    return data["shap_values"], data["X_sample"], float(data["expected_value"][0]), feat_names

@st.cache_data
def load_feature_names():
    with open(MODELS_DIR / "feature_names.json") as f:
        return json.load(f)

@st.cache_data
def load_raw_df():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    return df

@st.cache_data
def load_test_data():
    d = np.load(MODELS_DIR / "test_data.npz", allow_pickle=True)
    return d["X_test"], d["y_test"]


# ─────────────────────────────────────────────
# PREPROCESSING HELPER (for interactive input)
# ─────────────────────────────────────────────
def preprocess_input(raw_dict: dict, preprocessor, feature_names: list) -> pd.DataFrame:
    """Convert user input dict → preprocessed DataFrame ready for model."""
    df_input = pd.DataFrame([raw_dict])
    X_proc = preprocessor.transform(df_input)
    return pd.DataFrame(X_proc, columns=feature_names)


# ─────────────────────────────────────────────
# LOAD ALL ASSETS
# ─────────────────────────────────────────────
models       = load_models()
preprocessor = load_preprocessor()
metrics_data = load_metrics()
mlp_history  = load_mlp_history()
mlp_hp       = load_mlp_hp()
shap_vals, X_shap_sample, shap_expected, shap_feat_names = load_shap_data()
feature_names = load_feature_names()
raw_df        = load_raw_df()

X_shap_df = pd.DataFrame(X_shap_sample, columns=shap_feat_names)
all_metrics   = metrics_data["metrics"]
roc_data      = metrics_data["roc_data"]
best_params   = metrics_data["best_params"]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/UW_W-Logo_RGB.svg/200px-UW_W-Logo_RGB.svg.png", width=80)
    st.title("Telco Churn Predictor")
    st.caption("Customer Churn Analysis")
    st.divider()
    st.markdown("**Dataset:** Telco Customer Churn")
    st.markdown("**Task:** Binary Classification")
    st.markdown("**7,043** customers · **20** features")
    st.divider()
    st.markdown("**Best Model (F1):**")
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    st.success(f"✅ {best_model_name}")
    st.metric("F1 Score", all_metrics[best_model_name]["f1"])
    st.metric("AUC-ROC",  all_metrics[best_model_name]["auc_roc"])

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Executive Summary",
    "📊 Descriptive Analytics",
    "🤖 Model Performance",
    "🔍 Explainability & Prediction",
])


# ══════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════
with tab1:

    # ── Hero Banner ─────────────────────────────────────────────────
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                padding: 40px 36px 32px 36px; border-radius: 16px; margin-bottom: 28px;">
        <div style="display:flex; align-items:center; gap:14px; margin-bottom:10px;">
            <span style="font-size:42px;">📡</span>
            <div>
                <h1 style="color:#ffffff; margin:0; font-size:2.1rem; font-weight:800;
                            letter-spacing:-0.5px;">Telco Customer Churn Prediction</h1>
                <p style="color:#a8c8f8; margin:4px 0 0 0; font-size:1rem;">
                    End-to-End Machine Learning Pipeline &nbsp;|&nbsp; Binary Classification
                </p>
            </div>
        </div>
        <p style="color:#d0e4ff; margin:14px 0 0 0; font-size:1.05rem; line-height:1.65;">
            An end-to-end machine learning pipeline to predict which telecom customers are at risk
            of churning — combining exploratory analysis, five tuned models, SHAP explainability,
            and an interactive prediction interface.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi_style = (
        "background:{bg}; border:1px solid {border}; border-radius:12px; "
        "padding:18px 16px; text-align:center;"
    )
    def kpi_card(col, icon, label, value, bg, border, val_color):
        col.markdown(f"""
        <div style="{kpi_style.format(bg=bg, border=border)}">
            <div style="font-size:1.8rem;">{icon}</div>
            <div style="font-size:1.55rem; font-weight:800; color:{val_color}; margin:4px 0 2px 0;">{value}</div>
            <div style="font-size:0.78rem; color:#555; font-weight:600; text-transform:uppercase;
                        letter-spacing:0.5px;">{label}</div>
        </div>""", unsafe_allow_html=True)

    kpi_card(kpi1, "👥", "Total Customers",   "7,043",   "#f0f7ff", "#bdd7f5", "#1565c0")
    kpi_card(kpi2, "⚠️", "Churned",           "1,869",   "#fff4f4", "#f5c6c6", "#c62828")
    kpi_card(kpi3, "📊", "Churn Rate",        "26.5%",   "#fff8e6", "#f5dfa0", "#e65100")
    kpi_card(kpi4, "🔢", "Features",          "20",      "#f3f0ff", "#c9baee", "#4527a0")
    kpi_card(kpi5, "🏆", f"Best F1 ({best_model_name.split()[0]})",
             f"{all_metrics[best_model_name]['f1']:.4f}", "#f0fff4", "#a8dfc0", "#1b5e20")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── About the Dataset ────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
        <span style="font-size:1.5rem;">🗂️</span>
        <h3 style="margin:0; color:#1a237e;">About the Dataset</h3>
    </div>
    """, unsafe_allow_html=True)

    col_desc, col_feat = st.columns([3, 2])
    with col_desc:
        st.markdown("""
        This project uses the **IBM Telco Customer Churn** dataset, sourced from Kaggle.
        It contains information about **7,043 customers** of a fictional telecommunications
        company in California — capturing each customer's demographic profile, the services
        they subscribe to, and their account details.

        The **prediction target** is `Churn` — a binary variable indicating whether a customer
        left within the last month. Of the 7,043 customers, **1,869 (26.5%) churned**, making
        this a moderately imbalanced classification problem.
        """)

        st.markdown("""
        <div style="background:#fff8e1; border-left:5px solid #f9a825; border-radius:6px;
                    padding:14px 16px; margin-top:10px;">
            <b>⚖️ Handling Class Imbalance:</b> All tree-based and logistic models were trained
            with <code>class_weight='balanced'</code>; XGBoost used <code>scale_pos_weight</code>.
            Primary metrics are <b>F1 Score</b> and <b>AUC-ROC</b> — more informative than raw
            accuracy under imbalance.
        </div>
        """, unsafe_allow_html=True)

    with col_feat:
        st.markdown("""
        <div style="background:#f8f9ff; border:1px solid #dce3f5; border-radius:12px; padding:18px;">
            <p style="font-weight:700; color:#3949ab; margin:0 0 12px 0; font-size:0.95rem;">
                📋 Feature Categories
            </p>
            <div style="margin-bottom:10px;">
                <span style="background:#e3f2fd; color:#1565c0; padding:3px 9px; border-radius:20px;
                             font-size:0.82rem; font-weight:600;">👤 Demographics (4)</span>
                <p style="margin:6px 0 0 6px; font-size:0.85rem; color:#444;">
                    Gender, Senior Citizen, Partner, Dependents
                </p>
            </div>
            <div style="margin-bottom:10px;">
                <span style="background:#e8f5e9; color:#2e7d32; padding:3px 9px; border-radius:20px;
                             font-size:0.82rem; font-weight:600;">📶 Services (8)</span>
                <p style="margin:6px 0 0 6px; font-size:0.85rem; color:#444;">
                    Phone, Internet, Security, Backup, Tech Support, Streaming…
                </p>
            </div>
            <div>
                <span style="background:#fce4ec; color:#880e4f; padding:3px 9px; border-radius:20px;
                             font-size:0.82rem; font-weight:600;">💳 Account (8)</span>
                <p style="margin:6px 0 0 6px; font-size:0.85rem; color:#444;">
                    Tenure, Contract, Billing, Payment Method, Monthly & Total Charges
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Why It Matters ───────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
        <span style="font-size:1.5rem;">💡</span>
        <h3 style="margin:0; color:#1a237e;">Why This Problem Matters</h3>
    </div>
    """, unsafe_allow_html=True)

    w1, w2, w3 = st.columns(3)
    def why_card(col, icon, title, body, bg, accent):
        col.markdown(f"""
        <div style="background:{bg}; border-top:4px solid {accent}; border-radius:10px;
                    padding:18px 16px; height:100%;">
            <div style="font-size:1.6rem; margin-bottom:6px;">{icon}</div>
            <div style="font-weight:700; color:#222; margin-bottom:8px; font-size:0.97rem;">{title}</div>
            <div style="font-size:0.87rem; color:#444; line-height:1.6;">{body}</div>
        </div>""", unsafe_allow_html=True)

    why_card(w1, "💰", "Cost of Acquisition",
             "Acquiring a new customer costs <b>5–10× more</b> than retaining one. "
             "Proactive retention is far cheaper than win-back campaigns.",
             "#f0f7ff", "#1976d2")
    why_card(w2, "📈", "Revenue Protection",
             "Even a <b>1–2 pp reduction</b> in churn rate can save tens of millions "
             "in annual revenue for a company with millions of subscribers.",
             "#f3fff3", "#2e7d32")
    why_card(w3, "🎯", "Targeted Interventions",
             "SHAP explainability lets retention teams understand <b>why</b> a customer "
             "is at risk — enabling personalized, data-driven outreach.",
             "#fff8f0", "#e65100")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Approach ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
        <span style="font-size:1.5rem;">🔬</span>
        <h3 style="margin:0; color:#1a237e;">Our Approach</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#f9f9ff; border:1px solid #ddd; border-radius:12px; padding:20px 24px;">
        <div style="display:flex; flex-wrap:wrap; gap:8px; align-items:center;">
            <span style="background:#e3f2fd; border:1px solid #90caf9; border-radius:20px;
                         padding:5px 14px; font-size:0.88rem; font-weight:600; color:#1565c0;">
                1️⃣ EDA & Visualizations
            </span>
            <span style="color:#aaa; font-size:1.2rem;">→</span>
            <span style="background:#e8f5e9; border:1px solid #a5d6a7; border-radius:20px;
                         padding:5px 14px; font-size:0.88rem; font-weight:600; color:#2e7d32;">
                2️⃣ Preprocessing & Encoding
            </span>
            <span style="color:#aaa; font-size:1.2rem;">→</span>
            <span style="background:#fff3e0; border:1px solid #ffcc80; border-radius:20px;
                         padding:5px 14px; font-size:0.88rem; font-weight:600; color:#e65100;">
                3️⃣ 5 Models + GridSearchCV
            </span>
            <span style="color:#aaa; font-size:1.2rem;">→</span>
            <span style="background:#fce4ec; border:1px solid #f48fb1; border-radius:20px;
                         padding:5px 14px; font-size:0.88rem; font-weight:600; color:#880e4f;">
                4️⃣ SHAP Explainability
            </span>
            <span style="color:#aaa; font-size:1.2rem;">→</span>
            <span style="background:#ede7f6; border:1px solid #ce93d8; border-radius:20px;
                         padding:5px 14px; font-size:0.88rem; font-weight:600; color:#4527a0;">
                5️⃣ Interactive Deployment
            </span>
        </div>
        <p style="margin:14px 0 0 0; font-size:0.92rem; color:#444; line-height:1.65;">
            Five models were trained and evaluated: <b>Logistic Regression</b> (baseline),
            <b>Decision Tree</b>, <b>Random Forest</b>, <b>XGBoost</b>, and a
            <b>Multi-Layer Perceptron (MLP)</b>. All models were tuned via
            <b>5-fold cross-validation with GridSearchCV</b> using F1 as the scoring metric.
            A <b>70/30 stratified train/test split</b> was used throughout with
            <code>random_state=42</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Findings ─────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
        <span style="font-size:1.5rem;">🔑</span>
        <h3 style="margin:0; color:#1a237e;">Key Findings</h3>
    </div>
    """, unsafe_allow_html=True)

    findings = [
        ("#e3f2fd", "#1565c0", "#1976d2", "🏆",
         "Best Model: Random Forest (F1 = 0.6387)",
         "Random Forest outperformed all models on F1 score (0.6387), followed by XGBoost (0.6266) "
         "and Logistic Regression (0.6234). The MLP achieved the highest accuracy (0.7937) but "
         "lower recall, missing more actual churners."),
        ("#fff8e1", "#f57f17", "#f9a825", "⏱️",
         "Tenure is the #1 Churn Predictor",
         "Short-tenure customers churn at 3× the rate of long-tenured ones. SHAP confirms that "
         "low tenure is the single strongest driver pushing the prediction toward churn. "
         "New customers in their first 12 months are the highest-risk segment."),
        ("#fce4ec", "#880e4f", "#e91e63", "📄",
         "Contract Type Drives Massive Churn Differences",
         "Month-to-month customers churn at ~43% vs. only ~3% for two-year contracts. "
         "Converting flexible-plan customers to longer commitments is the highest-leverage "
         "retention lever available."),
        ("#e8f5e9", "#1b5e20", "#388e3c", "🌐",
         "Fiber Optic Customers Churn at 3× the Rate of DSL",
         "Despite paying more, fiber optic subscribers have dramatically higher churn, "
         "suggesting the product may not justify its premium price point or that "
         "competitors offer better alternatives in this segment."),
        ("#ede7f6", "#311b92", "#673ab7", "💳",
         "Electronic Check = Highest-Risk Payment Method (~45% churn)",
         "Electronic check users churn at nearly double the rate of automatic payment methods. "
         "This may proxy for lower customer engagement and commitment. "
         "Automatic payment enrollment could serve as both a signal and a retention tool."),
        ("#e0f7fa", "#004d40", "#00897b", "🔍",
         "SHAP Makes Every Prediction Explainable",
         "SHAP waterfall plots decompose each individual prediction into feature contributions, "
         "giving retention managers a transparent, actionable explanation — not just a black-box "
         "score — for why a specific customer is flagged as high-risk."),
    ]

    for i in range(0, len(findings), 2):
        c1, c2 = st.columns(2)
        for col, f in zip([c1, c2], findings[i:i+2]):
            bg, txt_c, border_c, icon, title, body = f
            col.markdown(f"""
            <div style="background:{bg}; border:1px solid {border_c}; border-left:5px solid {border_c};
                        border-radius:10px; padding:16px 18px; margin-bottom:12px; height:95%;">
                <div style="font-size:1.3rem; margin-bottom:6px;">{icon}</div>
                <div style="font-weight:700; color:{txt_c}; font-size:0.97rem;
                            margin-bottom:8px;">{title}</div>
                <div style="font-size:0.87rem; color:#333; line-height:1.65;">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Navigation hint ───────────────────────────────────────────────
    st.markdown("""
    <div style="background:#f0f4ff; border:1px solid #c5cfe8; border-radius:10px;
                padding:16px 20px; display:flex; align-items:center; gap:12px;">
        <span style="font-size:1.4rem;">🗺️</span>
        <div style="font-size:0.92rem; color:#333; line-height:1.6;">
            <b>Explore the full analysis:</b> &nbsp;
            📊 <b>Descriptive Analytics</b> — visualizations &amp; patterns &nbsp;|&nbsp;
            🤖 <b>Model Performance</b> — metrics, ROC curves &amp; hyperparameters &nbsp;|&nbsp;
            🔍 <b>Explainability &amp; Prediction</b> — SHAP plots &amp; live interactive prediction
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — DESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════
with tab2:
    st.title("📊 Descriptive Analytics")
    st.markdown("Deep exploration of the Telco Customer Churn dataset before modeling.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", "7,043")
    col2.metric("Features", "20")
    col3.metric("Numerical Features", "3")
    col4.metric("Categorical Features", "16")

    st.divider()

    # ── 1.1 Dataset Introduction ─────────────────────────────────────
    st.subheader("1.1 Dataset Overview")
    desc_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    st.dataframe(raw_df[desc_cols].describe().round(2), width="stretch")
    st.caption(
        "**Summary statistics for numerical features.** Tenure ranges from 0 to 72 months with "
        "an average of ~32 months, indicating a mix of very new and long-standing customers. "
        "Monthly charges range from \\$18.25 to \\$118.75, with an average of ~\\$64.76."
    )

    st.divider()

    # ── 1.2 Target Distribution ──────────────────────────────────────
    st.subheader("1.2 Target Variable Distribution (Churn)")

    churn_counts = raw_df["Churn"].value_counts().reset_index()
    churn_counts.columns = ["Churn", "Count"]
    churn_counts["Percentage"] = (churn_counts["Count"] / len(raw_df) * 100).round(1)

    fig_churn = px.bar(
        churn_counts, x="Churn", y="Count",
        color="Churn", text="Percentage",
        color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
        title="Churn Distribution: Retained vs. Churned Customers",
        labels={"Count": "Number of Customers"},
    )
    fig_churn.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_churn.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_churn, width="stretch")
    st.markdown("""
    <div style="background:#fff4f4; border:1px solid #f5c6c6; border-left:5px solid #e53935;
                border-radius:8px; padding:14px 18px; margin-top:4px;">
        <b>📌 Insight — Target Class Distribution:</b> Out of 7,043 customers,
        <b>5,174 (73.5%) were retained</b> and <b>1,869 (26.5%) churned</b> within the last month.
        This moderate class imbalance means that a naive "predict no churn" baseline would achieve
        73.5% accuracy, so we use <b>F1 and AUC-ROC</b> as the primary evaluation metrics, and
        apply <b>class weighting</b> during model training to avoid the model ignoring the minority class.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── 1.3 Feature Distributions & Relationships ────────────────────
    st.subheader("1.3 Feature Distributions and Relationships (≥4 Visualizations)")

    # Viz 1: Tenure by Churn (violin)
    st.markdown("#### Visualization 1: Tenure Distribution by Churn Status")
    fig_tenure = px.violin(
        raw_df, x="Churn", y="tenure", color="Churn", box=True, points=False,
        color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
        title="Customer Tenure (months) by Churn Status",
        labels={"tenure": "Tenure (months)"},
    )
    fig_tenure.update_layout(showlegend=False, height=420)
    st.plotly_chart(fig_tenure, width="stretch")
    st.markdown("""
    <div style="background:#e8f5e9; border:1px solid #a5d6a7; border-left:5px solid #2e7d32;
                border-radius:8px; padding:14px 18px; margin-top:4px;">
        <b>📌 Insight — Tenure vs. Churn:</b> The median tenure for churned customers is
        <b>~10 months</b>, compared to <b>~38 months</b> for retained customers — a nearly 4× gap.
        Churn risk is highest in the <b>first 12 months</b> of service. This makes tenure the
        single most powerful numerical predictor: a customer who has stayed for 5+ years is
        far less likely to leave than someone still evaluating the service.
    </div>
    """, unsafe_allow_html=True)

    # Viz 2: Monthly Charges by Internet Service and Churn
    st.markdown("#### Visualization 2: Monthly Charges by Internet Service Type and Churn")
    fig_mc = px.box(
        raw_df, x="InternetService", y="MonthlyCharges", color="Churn",
        color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
        title="Monthly Charges by Internet Service Type and Churn Status",
        labels={"MonthlyCharges": "Monthly Charges ($)"},
    )
    fig_mc.update_layout(height=440)
    st.plotly_chart(fig_mc, width="stretch")
    st.markdown("""
    <div style="background:#fff8e1; border:1px solid #ffe082; border-left:5px solid #f9a825;
                border-radius:8px; padding:14px 18px; margin-top:4px;">
        <b>📌 Insight — Internet Service &amp; Charges:</b> Fiber optic customers pay
        <b>~$80–90/month on average</b> and have a dramatically higher churn rate compared to DSL
        customers at similar price points. This suggests fiber optic subscribers feel the product
        does <b>not justify its premium price</b>, or that competitive alternatives are more
        attractive in that segment — making this a key product quality signal.
    </div>
    """, unsafe_allow_html=True)

    # Viz 3: Churn rate by Contract Type
    st.markdown("#### Visualization 3: Churn Rate by Contract Type")
    contract_churn = raw_df.groupby("Contract")["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).reset_index()
    contract_churn.columns = ["Contract", "Churn Rate (%)"]
    contract_churn = contract_churn.sort_values("Churn Rate (%)", ascending=False)

    fig_contract = px.bar(
        contract_churn, x="Contract", y="Churn Rate (%)",
        color="Churn Rate (%)", text=contract_churn["Churn Rate (%)"].round(1),
        color_continuous_scale="RdYlGn_r",
        title="Churn Rate (%) by Contract Type",
    )
    fig_contract.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_contract.update_layout(coloraxis_showscale=False, height=420)
    st.plotly_chart(fig_contract, width="stretch")
    st.markdown("""
    <div style="background:#e3f2fd; border:1px solid #90caf9; border-left:5px solid #1565c0;
                border-radius:8px; padding:14px 18px; margin-top:4px;">
        <b>📌 Insight — Contract Type:</b> Month-to-month customers churn at <b>~43%</b> —
        more than <b>14× higher</b> than two-year contract holders (~3%). This is the
        strongest categorical predictor of churn. Customers on flexible monthly plans face
        no exit barriers, while long-term contract holders are financially committed.
        <b>Converting month-to-month customers to annual plans</b> is the single highest-leverage
        retention strategy available.
    </div>
    """, unsafe_allow_html=True)

    # Viz 4: Churn rate by Payment Method
    st.markdown("#### Visualization 4: Churn Rate by Payment Method")
    pay_churn = raw_df.groupby("PaymentMethod")["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).reset_index()
    pay_churn.columns = ["PaymentMethod", "Churn Rate (%)"]
    pay_churn = pay_churn.sort_values("Churn Rate (%)", ascending=False)

    fig_pay = px.bar(
        pay_churn, x="PaymentMethod", y="Churn Rate (%)",
        color="Churn Rate (%)", text=pay_churn["Churn Rate (%)"].round(1),
        color_continuous_scale="Reds",
        title="Churn Rate (%) by Payment Method",
    )
    fig_pay.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_pay.update_layout(coloraxis_showscale=False, height=420)
    st.plotly_chart(fig_pay, width="stretch")
    st.markdown("""
    <div style="background:#fce4ec; border:1px solid #f48fb1; border-left:5px solid #c2185b;
                border-radius:8px; padding:14px 18px; margin-top:4px;">
        <b>📌 Insight — Payment Method:</b> Electronic check users churn at <b>~45%</b> —
        nearly <b>3× higher</b> than automatic credit card (~15%) or bank transfer (~17%) users.
        Customers who pay manually may already be disengaged or planning to leave. Automatic
        payment enrollment acts as both a <b>commitment signal</b> and a practical retention tool,
        since it reduces friction to stay.
    </div>
    """, unsafe_allow_html=True)

    # Viz 5: Senior Citizen churn rates
    st.markdown("#### Visualization 5: Churn Rate by Senior Citizen Status and Partner/Dependent Combo")
    raw_df["SeniorLabel"] = raw_df["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
    senior_churn = raw_df.groupby(["SeniorLabel", "Partner"])["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).reset_index()
    senior_churn.columns = ["Senior Status", "Partner", "Churn Rate (%)"]

    fig_senior = px.bar(
        senior_churn, x="Senior Status", y="Churn Rate (%)", color="Partner",
        barmode="group", text=senior_churn["Churn Rate (%)"].round(1),
        title="Churn Rate by Senior Citizen Status and Partner",
        color_discrete_map={"Yes": "#3498db", "No": "#e67e22"},
    )
    fig_senior.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_senior.update_layout(height=430)
    st.plotly_chart(fig_senior, width="stretch")
    st.markdown("""
    <div style="background:#ede7f6; border:1px solid #ce93d8; border-left:5px solid #6a1b9a;
                border-radius:8px; padding:14px 18px; margin-top:4px;">
        <b>📌 Insight — Demographics:</b> Senior citizens without a partner churn at <b>~53%</b>,
        while non-senior customers with a partner churn at only <b>~17%</b>. Having a partner is
        a strong <b>protective factor</b> — likely because household/family plans create stickiness.
        Senior solo customers may be more price-sensitive or switching to simpler plans.
        This demographic insight can help <b>target retention outreach</b> more precisely.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── 1.4 Correlation Heatmap ──────────────────────────────────────
    st.subheader("1.4 Correlation Heatmap")
    churn_num = raw_df.copy()
    churn_num["Churn_binary"] = (churn_num["Churn"] == "Yes").astype(int)
    corr_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn_binary"]
    corr_matrix = churn_num[corr_cols].corr()

    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Heatmap (Numerical Features + Churn)",
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, width="stretch")
    st.markdown("""
    <div style="background:#e0f7fa; border:1px solid #80deea; border-left:5px solid #00838f;
                border-radius:8px; padding:14px 18px; margin-top:4px;">
        <b>📌 Insight — Correlations:</b>
        <ul style="margin:8px 0 0 0; padding-left:18px; line-height:1.75;">
            <li><b>Tenure ↔ Churn: −0.35</b> — The strongest signal. Longer-tenured customers are
            much less likely to leave.</li>
            <li><b>MonthlyCharges ↔ Churn: +0.19</b> — Higher-paying customers churn more,
            suggesting value perception issues.</li>
            <li><b>Tenure ↔ TotalCharges: +0.83</b> — High multicollinearity (expected, since
            TotalCharges ≈ tenure × monthly rate). Tree-based models handle this naturally; linear
            models benefit from dropping one.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════
with tab3:
    st.title("🤖 Model Performance")
    st.markdown("All five models evaluated on the held-out test set (30% of data, 2,113 customers).")

    # ── 2.7 Summary Table ────────────────────────────────────────────
    st.subheader("2.7 Model Comparison Summary")

    metrics_df = pd.DataFrame(all_metrics).T.reset_index()
    metrics_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    metrics_df = metrics_df.sort_values("F1 Score", ascending=False)

    def highlight_best(s):
        is_best = s == s.max()
        return ["background-color: #d4edda; font-weight: bold" if v else "" for v in is_best]

    styled = metrics_df.style.apply(highlight_best, subset=["Accuracy", "F1 Score", "AUC-ROC"])
    st.dataframe(styled, width="stretch", hide_index=True)

    # Bar chart F1
    fig_f1 = px.bar(
        metrics_df.sort_values("F1 Score"),
        x="F1 Score", y="Model", orientation="h",
        color="F1 Score", color_continuous_scale="Blues",
        title="F1 Score Comparison (Test Set)",
        text=metrics_df.sort_values("F1 Score")["F1 Score"].round(4),
    )
    fig_f1.update_traces(textposition="outside")
    fig_f1.update_layout(coloraxis_showscale=False, height=380)
    st.plotly_chart(fig_f1, width="stretch")

    # Bar chart AUC
    fig_auc = px.bar(
        metrics_df.sort_values("AUC-ROC"),
        x="AUC-ROC", y="Model", orientation="h",
        color="AUC-ROC", color_continuous_scale="Greens",
        title="AUC-ROC Comparison (Test Set)",
        text=metrics_df.sort_values("AUC-ROC")["AUC-ROC"].round(4),
    )
    fig_auc.update_traces(textposition="outside")
    fig_auc.update_layout(coloraxis_showscale=False, height=380)
    st.plotly_chart(fig_auc, width="stretch")

    st.markdown("""
**Model Analysis:**
Random Forest achieved the best F1 score (0.6387), followed closely by XGBoost (0.6266) and Logistic
Regression (0.6234). AUC-ROC scores are more tightly clustered (0.83–0.84), suggesting all models
have similar ranking ability. The MLP had the highest accuracy (0.7937) but lower F1 (0.5808) due to
lower recall — it predicts fewer customers as churners, missing actual churners more often.
The trade-off is clear: tree-based models with class balancing better prioritize recall for the minority
churn class, while the MLP optimizes overall accuracy. For a churn use case, F1/recall are more important
since false negatives (missed churners) are costly.
    """)

    st.divider()

    # ── Best Hyperparameters ─────────────────────────────────────────
    st.subheader("Best Hyperparameters per Model")
    for model_name, params in best_params.items():
        with st.expander(f"🔧 {model_name}"):
            st.json(params)

    st.divider()

    # ── ROC Curves ──────────────────────────────────────────────────
    st.subheader("ROC Curves (All Models)")
    fig_roc = go.Figure()
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for (model_name, roc), color in zip(roc_data.items(), colors):
        auc = all_metrics[model_name]["auc_roc"]
        fig_roc.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"],
            mode="lines", name=f"{model_name} (AUC={auc:.4f})",
            line=dict(color=color, width=2),
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Baseline", line=dict(color="gray", dash="dash", width=1),
    ))
    fig_roc.update_layout(
        title="ROC Curves — All Models vs. Random Baseline",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        legend=dict(x=0.6, y=0.1),
    )
    st.plotly_chart(fig_roc, width="stretch")
    st.caption(
        "**All models significantly outperform the random baseline (diagonal dashed line) with "
        "AUC-ROC scores ranging from 0.83 to 0.84.** The tight clustering of ROC curves indicates "
        "that the models have comparable discriminative ability — the main differences emerge in the "
        "precision-recall trade-off governed by the classification threshold. Logistic Regression, "
        "despite being the simplest model, achieves competitive AUC-ROC (0.8444), suggesting that "
        "the churn signal in this dataset is largely linearly separable after preprocessing."
    )

    st.divider()

    # ── Model-Specific Plots ─────────────────────────────────────────
    st.subheader("Model-Specific Details")

    subtab_lr, subtab_dt, subtab_rf, subtab_xgb, subtab_mlp = st.tabs([
        "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "MLP Neural Network"
    ])

    with subtab_lr:
        st.markdown("**Logistic Regression Baseline**")
        m = all_metrics["Logistic Regression"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  m["accuracy"])
        c2.metric("Precision", m["precision"])
        c3.metric("Recall",    m["recall"])
        c4.metric("F1 Score",  m["f1"])
        c5.metric("AUC-ROC",   m["auc_roc"])
        st.info("Parameters: C=1.0, solver=lbfgs, class_weight=balanced, max_iter=1000")
        st.markdown(
            "Logistic Regression serves as a strong interpretable baseline. "
            "It achieves the second-highest AUC-ROC (0.8444) and competitive F1 (0.6234), "
            "benefiting from the balanced class weights. Its recall (0.7968) is high, "
            "meaning it catches ~80% of actual churners, which is valuable for a retention campaign."
        )

    with subtab_dt:
        st.markdown("**Decision Tree (5-fold CV GridSearchCV)**")
        m = all_metrics["Decision Tree"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  m["accuracy"])
        c2.metric("Precision", m["precision"])
        c3.metric("Recall",    m["recall"])
        c4.metric("F1 Score",  m["f1"])
        c5.metric("AUC-ROC",   m["auc_roc"])
        st.info("Best params from GridSearch: max_depth=7, min_samples_leaf=50")
        st.markdown(
            "The Decision Tree was tuned over max_depth ∈ {3,5,7,10} and min_samples_leaf ∈ {5,10,20,50} "
            "using 5-fold CV. Best depth=7 captures meaningful interactions, while min_samples_leaf=50 "
            "prevents overfitting on this 4,930-sample training set. "
            "Its AUC-ROC (0.8308) is slightly lower than ensemble methods, as expected."
        )

    with subtab_rf:
        st.markdown("**Random Forest (5-fold CV GridSearchCV) — Best F1 Model**")
        m = all_metrics["Random Forest"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  m["accuracy"])
        c2.metric("Precision", m["precision"])
        c3.metric("Recall",    m["recall"])
        c4.metric("F1 Score",  m["f1"])
        c5.metric("AUC-ROC",   m["auc_roc"])
        st.success("✅ Best F1 Score: 0.6387")
        st.info("Best params: n_estimators=200, max_depth=8")
        st.markdown(
            "Random Forest with 200 trees and max_depth=8 achieved the highest F1 score (0.6387). "
            "The ensemble reduces variance compared to a single decision tree and benefits from "
            "diverse trees trained on bootstrapped samples. Depth=8 gives enough complexity to "
            "capture feature interactions without overfitting."
        )

    with subtab_xgb:
        st.markdown("**XGBoost (5-fold CV GridSearchCV)**")
        m = all_metrics["XGBoost"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  m["accuracy"])
        c2.metric("Precision", m["precision"])
        c3.metric("Recall",    m["recall"])
        c4.metric("F1 Score",  m["f1"])
        c5.metric("AUC-ROC",   m["auc_roc"])
        st.info("Best params: learning_rate=0.1, max_depth=3, n_estimators=50")
        st.markdown(
            "XGBoost used scale_pos_weight to address class imbalance and was tuned over "
            "n_estimators ∈ {50,100,200}, max_depth ∈ {3,4,5}, and learning_rate ∈ {0.05,0.1}. "
            "The best model used shallow trees (depth=3) with a moderate learning rate, "
            "achieving AUC-ROC of 0.8437 and the highest recall (0.8093) among all models."
        )

    with subtab_mlp:
        st.markdown("**MLP Neural Network (sklearn MLPClassifier)**")
        m = all_metrics["MLP Neural Network"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  m["accuracy"])
        c2.metric("Precision", m["precision"])
        c3.metric("Recall",    m["recall"])
        c4.metric("F1 Score",  m["f1"])
        c5.metric("AUC-ROC",   m["auc_roc"])
        st.info("Architecture: Input → Dense(128, ReLU) → Dense(128, ReLU) → Sigmoid Output")
        st.info("Optimizer: Adam, Loss: Binary Cross-Entropy (log_loss), Early Stopping enabled")

        # Training history
        st.markdown("##### Training History")
        history_df = pd.DataFrame({
            "Epoch":     list(range(1, len(mlp_history["loss_curve"]) + 1)),
            "Train Loss": mlp_history["loss_curve"],
            "Val Accuracy": mlp_history["val_scores"],
        })
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=history_df["Epoch"], y=history_df["Train Loss"],
            name="Training Loss", mode="lines", line=dict(color="#e74c3c"),
        ))
        fig_hist2 = go.Figure()
        fig_hist2.add_trace(go.Scatter(
            x=history_df["Epoch"], y=history_df["Val Accuracy"],
            name="Validation Accuracy", mode="lines", line=dict(color="#3498db"),
        ))
        c_left, c_right = st.columns(2)
        with c_left:
            fig_hist.update_layout(title="MLP Training Loss", xaxis_title="Epoch",
                                   yaxis_title="Loss", height=350)
            st.plotly_chart(fig_hist, width="stretch")
        with c_right:
            fig_hist2.update_layout(title="MLP Validation Accuracy", xaxis_title="Epoch",
                                    yaxis_title="Accuracy", height=350)
            st.plotly_chart(fig_hist2, width="stretch")
        st.caption(
            "**The MLP converges rapidly (early stopping triggered) with training loss decreasing "
            "steadily.** Validation accuracy plateaus around 0.79–0.80, consistent with the test set "
            "accuracy. The model achieves the highest overall accuracy but trades off recall for "
            "precision — it makes fewer false positive predictions of churn but misses more actual churners."
        )

        # BONUS: HP Tuning results
        st.markdown("##### ⭐ BONUS: MLP Hyperparameter Tuning Results")
        st.success(f"Best params: {mlp_hp['best_params']} | Best CV F1: {mlp_hp['best_f1_cv']}")
        hp_df = pd.DataFrame({
            "Configuration": mlp_hp["cv_results"]["params"],
            "Mean CV F1":    mlp_hp["cv_results"]["mean_test_score"],
            "Std CV F1":     mlp_hp["cv_results"]["std_test_score"],
        }).sort_values("Mean CV F1", ascending=False)
        fig_hp = px.bar(
            hp_df.head(10), x="Mean CV F1", y="Configuration", orientation="h",
            error_x="Std CV F1",
            color="Mean CV F1", color_continuous_scale="Blues",
            title="Top 10 MLP Configurations by CV F1 Score",
        )
        fig_hp.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig_hp, width="stretch")
        st.caption(
            "**Grid search over hidden layer sizes {(64,64), (128,128), (64,128,64), (256,128)}, "
            "learning rates {0.001, 0.005, 0.01}, and L2 regularization (alpha) {0.0001, 0.001} "
            "with 3-fold CV.** The best configuration uses hidden_layer_sizes=(128,128), "
            "learning_rate_init=0.005, and alpha=0.0001, achieving CV F1=0.6078. This shows that "
            "deeper/wider architectures with moderate learning rate provide the best bias-variance "
            "trade-off for this dataset."
        )


# ══════════════════════════════════════════════
# TAB 4 — EXPLAINABILITY & INTERACTIVE PREDICTION
# ══════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 60%, #415a77 100%);
                padding: 32px 36px 26px 36px; border-radius: 14px; margin-bottom: 24px;">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
            <span style="font-size:2rem;">🔍</span>
            <h2 style="color:#fff; margin:0; font-size:1.7rem; font-weight:800;">
                Explainability & Interactive Prediction
            </h2>
        </div>
        <p style="color:#c8d8e8; margin:0; font-size:0.97rem; line-height:1.6;">
            Understand <em>why</em> the model makes each prediction using SHAP values,
            then try your own customer profile and get an instant churn probability
            with a gauge and profile summary — all in one place.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # SECTION A — SHAP ANALYSIS
    # ════════════════════════════════════════════
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
        <span style="font-size:1.4rem;">📊</span>
        <h3 style="margin:0; color:#1a237e;">Part A — SHAP Analysis (XGBoost)</h3>
    </div>
    <p style="color:#555; font-size:0.9rem; margin:0 0 16px 0;">
        SHAP (SHapley Additive exPlanations) decomposes every prediction into individual
        feature contributions — showing both <strong>magnitude</strong> and
        <strong>direction</strong> of impact.
    </p>
    """, unsafe_allow_html=True)

    # Plot 1 + Plot 2 side-by-side
    shap_col1, shap_col2 = st.columns(2)

    with shap_col1:
        st.markdown("##### 🐝 Beeswarm — Feature Impact Distribution")
        fig_bee, _ = plt.subplots(figsize=(7, 6))
        shap.summary_plot(shap_vals, X_shap_df, show=False, max_display=12, plot_type="dot")
        plt.tight_layout()
        st.pyplot(fig_bee, clear_figure=True)
        plt.close(fig_bee)
        st.markdown("""
        <div style="background:#e8f5e9; border-left:4px solid #2e7d32; border-radius:6px;
                    padding:10px 14px; font-size:0.85rem; color:#333; margin-top:6px;">
            Each dot = one customer. <b>Right = increases churn risk</b>, left = decreases it.
            Red = high feature value, blue = low. <b>Short tenure (blue, right)</b> is the
            strongest churn driver; a <b>two-year contract (red, left)</b> strongly suppresses it.
        </div>
        """, unsafe_allow_html=True)

    with shap_col2:
        st.markdown("##### 📊 Bar — Mean Absolute Feature Importance")
        fig_bar, _ = plt.subplots(figsize=(7, 6))
        shap.summary_plot(shap_vals, X_shap_df, show=False, max_display=12, plot_type="bar")
        plt.tight_layout()
        st.pyplot(fig_bar, clear_figure=True)
        plt.close(fig_bar)
        st.markdown("""
        <div style="background:#e3f2fd; border-left:4px solid #1565c0; border-radius:6px;
                    padding:10px 14px; font-size:0.85rem; color:#333; margin-top:6px;">
            Ranks features by average absolute SHAP value. <b>Tenure, contract type, and
            monthly charges</b> dominate. <b>OnlineSecurity and TechSupport</b> also rank
            highly — customers without these add-ons are at elevated risk.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plot 3: Waterfall for a pre-loaded sample
    st.markdown("##### 🌊 Waterfall — Single Customer Explanation")
    st.caption("Select a pre-loaded example to see how each feature contributes to that customer's prediction.")

    sample_options = {
        "⚠️ High-Risk Churner (sample 0)": 0,
        "✅ Low-Risk Customer  (sample 1)": 1,
        "🟡 Medium-Risk Customer (sample 2)": 2,
    }
    wf_col1, wf_col2 = st.columns([1, 3])
    with wf_col1:
        selected_sample_label = st.selectbox("Pre-loaded sample:", list(sample_options.keys()))
    sample_idx = sample_options[selected_sample_label]

    fig_wf, _ = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[sample_idx],
            base_values=shap_expected,
            data=X_shap_df.iloc[sample_idx].values,
            feature_names=shap_feat_names,
        ),
        show=False, max_display=12,
    )
    plt.tight_layout()
    st.pyplot(fig_wf, clear_figure=True)
    plt.close(fig_wf)
    st.markdown("""
    <div style="background:#fff8e1; border-left:4px solid #f9a825; border-radius:6px;
                padding:10px 14px; font-size:0.85rem; color:#333; margin-top:6px;">
        <b>How to read this:</b> Start from the base value (average prediction). Each bar
        <span style="color:#e53935;"><b>pushes up</b></span> (red) or
        <span style="color:#1565c0;"><b>pushes down</b></span> (blue) toward the final churn
        probability. The combination of all SHAP values lands on the model's output for this customer.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ════════════════════════════════════════════
    # SECTION B — INTERACTIVE PREDICTION
    # ════════════════════════════════════════════
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
        <span style="font-size:1.4rem;">🎛️</span>
        <h3 style="margin:0; color:#1a237e;">Part B — Interactive Prediction</h3>
    </div>
    <p style="color:#555; font-size:0.9rem; margin:0 0 16px 0;">
        Build your own customer profile using the controls below.
        Choose any trained model, click <strong>Predict Churn</strong>, and instantly see
        the predicted class, churn probability, and a customer profile summary.
    </p>
    """, unsafe_allow_html=True)

    # Model selector — prominent, full-width styled box
    st.markdown("""
    <div style="background:#f0f4ff; border:1px solid #c5cfe8; border-radius:10px;
                padding:14px 18px; margin-bottom:16px;">
        <b>🤖 Step 1 — Choose a model for prediction</b>
    </div>
    """, unsafe_allow_html=True)
    pred_model_name = st.selectbox(
        "Model:",
        list(models.keys()),
        index=list(models.keys()).index("Random Forest"),
        label_visibility="collapsed",
    )
    pred_model = models[pred_model_name]

    # Feature inputs in a styled card
    st.markdown("""
    <div style="background:#f9faff; border:1px solid #dce3f5; border-radius:10px;
                padding:14px 18px; margin-bottom:4px;">
        <b>👤 Step 2 — Set customer features</b>
        <span style="color:#777; font-size:0.83rem; margin-left:8px;">
            (8 key inputs shown · remaining features use dataset averages)
        </span>
    </div>
    """, unsafe_allow_html=True)

    inp_c1, inp_c2, inp_c3 = st.columns(3)
    with inp_c1:
        st.markdown("**📅 Account Details**")
        tenure_in = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges_in = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total_charges_in = st.number_input(
            "Total Charges ($)", min_value=0.0, max_value=9000.0,
            value=float(round(tenure_in * monthly_charges_in, 2)), step=10.0,
        )
    with inp_c2:
        st.markdown("**📄 Plan & Service**")
        contract_in = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_in = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment_in  = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
    with inp_c3:
        st.markdown("**🔒 Add-on Services & Demographics**")
        online_sec_in   = st.selectbox("Online Security",  ["No", "Yes", "No internet service"])
        tech_support_in = st.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
        senior_in       = st.selectbox("Senior Citizen",   ["No", "Yes"])

    # Build full input row (remaining features use typical defaults)
    input_dict = {
        "gender":           "Male",
        "SeniorCitizen":    1 if senior_in == "Yes" else 0,
        "Partner":          "No",
        "Dependents":       "No",
        "tenure":           tenure_in,
        "PhoneService":     "Yes",
        "MultipleLines":    "No",
        "InternetService":  internet_in,
        "OnlineSecurity":   online_sec_in,
        "OnlineBackup":     "No",
        "DeviceProtection": "No",
        "TechSupport":      tech_support_in,
        "StreamingTV":      "No",
        "StreamingMovies":  "No",
        "Contract":         contract_in,
        "PaperlessBilling": "Yes",
        "PaymentMethod":    payment_in,
        "MonthlyCharges":   monthly_charges_in,
        "TotalCharges":     total_charges_in,
    }

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮  Predict Churn", type="primary", use_container_width=True)

    if predict_btn:
        X_input    = preprocess_input(input_dict, preprocessor, feature_names)
        pred_class = int(pred_model.predict(X_input)[0])
        pred_prob  = float(pred_model.predict_proba(X_input)[0][1])

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Result banner ──────────────────────────────────────────
        if pred_class == 1:
            st.markdown(f"""
            <div style="background:#fff0f0; border:2px solid #e53935; border-radius:12px;
                        padding:20px 24px; margin-bottom:16px; display:flex;
                        align-items:center; gap:18px;">
                <span style="font-size:2.8rem;">⚠️</span>
                <div>
                    <div style="font-size:1.4rem; font-weight:800; color:#c62828;">
                        CHURN PREDICTED
                    </div>
                    <div style="font-size:1rem; color:#555; margin-top:4px;">
                        This customer has a <b style="color:#c62828;">{pred_prob:.1%} churn probability</b>
                        according to <b>{pred_model_name}</b>.
                        Consider proactive retention outreach.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#f0fff4; border:2px solid #2e7d32; border-radius:12px;
                        padding:20px 24px; margin-bottom:16px; display:flex;
                        align-items:center; gap:18px;">
                <span style="font-size:2.8rem;">✅</span>
                <div>
                    <div style="font-size:1.4rem; font-weight:800; color:#1b5e20;">
                        NO CHURN PREDICTED
                    </div>
                    <div style="font-size:1rem; color:#555; margin-top:4px;">
                        This customer has a <b style="color:#1b5e20;">{pred_prob:.1%} churn probability</b>
                        according to <b>{pred_model_name}</b>. Customer appears stable.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Gauge + feature summary ────────────────────────────────
        res_c1, res_c2 = st.columns([1, 1])
        with res_c1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_prob * 100,
                title={"text": "Churn Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#e53935" if pred_class == 1 else "#2e7d32"},
                    "steps": [
                        {"range": [0,  30], "color": "#e8f5e9"},
                        {"range": [30, 60], "color": "#fff9c4"},
                        {"range": [60,100], "color": "#ffebee"},
                    ],
                    "threshold": {
                        "line": {"color": "#333", "width": 3},
                        "thickness": 0.8, "value": 50,
                    },
                },
                number={"suffix": "%", "font": {"size": 38}},
            ))
            fig_gauge.update_layout(height=270, margin=dict(t=40, b=10))
            st.plotly_chart(fig_gauge, width="stretch")

        with res_c2:
            st.markdown("**📋 Customer Profile Summary**")
            summary_data = {
                "Feature": [
                    "Tenure", "Monthly Charges", "Total Charges",
                    "Contract", "Internet Service",
                    "Payment Method", "Online Security", "Tech Support", "Senior Citizen",
                ],
                "Value": [
                    f"{tenure_in} months", f"${monthly_charges_in:.2f}", f"${total_charges_in:.2f}",
                    contract_in, internet_in,
                    payment_in, online_sec_in, tech_support_in, senior_in,
                ],
            }
            st.dataframe(
                pd.DataFrame(summary_data),
                hide_index=True, width="stretch",
            )

