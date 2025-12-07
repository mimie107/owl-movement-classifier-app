import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Owl Movement Classifier", page_icon="🦉", layout="wide")

st.title("🦉 Owl Movement Classifier — XGBoost + SHAP + RAG")


# =============================================
# 1 — LOAD MODEL
# =============================================
@st.cache_resource
def load_model():
    return joblib.load("xgb_classifier.pkl")

clf = load_model()


# =============================================
# 2 — FEATURES
# =============================================
FEATURES = [
    "snr", "sigsd", "noise", "burstSlop",
    "snr_lag1", "snr_lag2",
    "sigsd_lag1", "noise_lag1",
    "snr_roll3", "noise_roll3",
    "hour_sin", "hour_cos",
    "day", "month"
]


# =============================================
# 3 — DATA UPLOAD
# =============================================
st.sidebar.header("📂 Upload Dataset")
uploaded = st.sidebar.file_uploader("Upload your processed df", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Dataset loaded!")
else:
    st.warning("Upload a processed dataset to continue.")
    st.stop()


# =============================================
# 4 — SHAP EXPLAINER
# =============================================
@st.cache_resource
def load_shap():
    return shap.TreeExplainer(clf)

explainer = load_shap()

def shap_for_row(row):
    X = row[FEATURES].values.reshape(1, -1)
    shap_vals = explainer.shap_values(X)

    if isinstance(shap_vals, np.ndarray):
        return shap_vals[0]
    if isinstance(shap_vals, list):
        return shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]

    return np.zeros(len(FEATURES))


def shap_text_summary(shap_vals):
    pairs = list(zip(FEATURES, shap_vals))
    top = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:5]

    text = "Top SHAP contributors:\n"
    for feat, val in top:
        direction = "↑ increases movement probability" if val > 0 else "↓ reduces movement probability"
        text += f"- {feat}: {val:.3f} → {direction}\n"
    return text


# =============================================
# 5 — RAG (NO TRANSFORMERS)
# =============================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()


rag_docs = {
    "movement_definition": """
Movement_class = 1 indicates a long gap in detections suggesting temporary
departure, vagrancy, or possible migratory behavior.
""",

    "feature_info": """
Important predictors include SNR, noise, lag features, rolling averages,
hour_sin/hour_cos, and signal stability indicators.
""",

    "xgboost_info": """
This classifier uses engineered features derived from your pipeline to detect
changes in pattern consistency and detection gaps.
"""
}

rag_embeddings = {
    k: embedder.encode(v, convert_to_tensor=True)
    for k, v in rag_docs.items()
}

def retrieve_context(query):
    q = embedder.encode(query, convert_to_tensor=True)
    sims = {k: util.cos_sim(q, emb).item() for k, emb in rag_embeddings.items()}
    ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    return "\n\n".join([rag_docs[k] for k, _ in ranked[:2]])


# =============================================
# SIMPLE RAG EXPLANATION (NO LLM)
# =============================================
def simple_rag_explanation(shap_text, rag_context):
    return f"""
### Explanation Summary

**SHAP-Based Feature Impact:**

{shap_text}

**Biological + Modeling Context (RAG):**

{rag_context}

The model detected patterns in the input row that match known signatures of
movement behavior (e.g., signal disruption, increased noise, lag instability).
"""


# =============================================
# SIDEBAR NAV
# =============================================
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Prediction Explorer", "🧠 RAG Explanation"]
)


# =============================================
# PAGE 1 — HOME
# =============================================
if page == "🏠 Home":
    st.header("Welcome")
    st.write("""
This is the **final XGBoost-based movement classifier**, aligned with your notebook.

Includes:
- XGBoost predictions  
- SHAP local explanations  
- RAG scientific context  
(No FLAN-T5 due to Streamlit Cloud restrictions)
""")


# =============================================
# PAGE 2 — PREDICTION EXPLORER
# =============================================
elif page == "🔍 Prediction Explorer":
    st.header("🔍 Prediction Explorer")

    idx = st.number_input("Row index:", min_value=0, max_value=len(df)-1, value=0)
    row = df.iloc[idx]

    # Prediction
    prob = clf.predict_proba([row[FEATURES]])[0, 1]
    pred = int(prob >= 0.30)

    st.subheader("Prediction")
    st.write(f"**Predicted class:** {'Movement (1)' if pred else 'Resident (0)'}")
    st.write(f"**Probability of movement:** {prob:.3f}")

    # SHAP
    shap_vals = shap_for_row(row)
    st.subheader("SHAP Explanation")
    st.text(shap_text_summary(shap_vals))

    # Waterfall
    shap_exp = shap.Explanation(
        values=shap_vals,
        base_values=explainer.expected_value,
        data=row[FEATURES].values,
        feature_names=FEATURES
    )

    fig = plt.figure(figsize=(9, 5))
    shap.plots.waterfall(shap_exp, show=False)
    st.pyplot(fig)


# =============================================
# PAGE 3 — RAG EXPLANATION
# =============================================
elif page == "🧠 RAG Explanation":
    st.header("🧠 RAG Explanation")

    idx = st.number_input("Row:", min_value=0, max_value=len(df)-1, value=0)
    row = df.iloc[idx]

    shap_vals = shap_for_row(row)
    shap_text = shap_text_summary(shap_vals)
    ctx = retrieve_context("movement prediction")

    explanation = simple_rag_explanation(shap_text, ctx)
    st.markdown(explanation)
