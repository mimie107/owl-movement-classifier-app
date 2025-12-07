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
    return joblib.load("classifier_model.pkl")

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
    ["🏠 Home", "📊 EDA Insights", "🔍 Prediction Explorer", "🧠 RAG Explanation"]
)



# =============================================
# PAGE 1 — HOME
# =============================================
if page == "🏠 Home":
    st.header("Welcome")
    st.write("""
Welcome to the **Owl Movement Classifier App**.

This tool helps us understand owl activity patterns at the Beaverhill Bird Observatory (BBO)
using detection data collected from the automated radio-telemetry tower.

### 🔍 What this app does
We analyze the detection signals collected from tagged owls and:
- Explore the dataset to understand detection patterns  
- Use a machine learning model (XGBoost) to classify whether an owl is **moving/migrating (1)** or **staying local/resident (0)**
- Explain *why* the model made a prediction using **SHAP feature interpretation**
- Provide **contextual scientific explanations** using a small Retrieval-Augmented Generation (RAG) system

### 🦉 Why this matters
These insights help us answer key ecological questions, such as:
- How long were owls detectable after tagging?
- When are owls most active (foraging vs flight times)?
- Do signal patterns suggest local movement or departure?
- How do signal strength, noise, and detection patterns change before migration?

### 📁 How to use the app
1. Upload your processed dataset in the sidebar.  
2. Navigate between pages:
   - **EDA Insights:** Explore patterns in detection times, SNR, noise, and class balance  
   - **Prediction Explorer:** View model predictions and SHAP explanations  
   - **RAG Explanation:** Get scientific context behind movement behavior  

This app brings together ecological knowledge and AI modeling to help us
identify migration behavior and better understand owl movement patterns over time.
""")

# =============================================
# PAGE 2 - EXPLORATORY DATA ANALYSIS (EDA)
# =============================================
elif page == "📊 EDA Insights":
    st.header("📊 Exploratory Data Analysis — Owl Detectability Insights")

    if uploaded is None:
        st.warning("Upload your dataset to explore.")
        st.stop()

    # ---------------------------
    # Dataset preview
    # ---------------------------
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    # Summary statistics
    # ---------------------------
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # ---------------------------
    # 1. Detection Times (Hourly Pattern)
    # ---------------------------
    if "hour" in df.columns:
        st.subheader("🕒 Detection Times (Hourly Pattern)")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["hour"], bins=24, color="skyblue", edgecolor="black")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Owl Detections by Hour")
        st.pyplot(fig)
        st.markdown("""
        **Insight:**  
        Owls are often detected around dusk/night. Peaks here may indicate foraging or early movement activity.
        """)

    # ---------------------------
    # 2. SNR distribution (distance proxy)
    # ---------------------------
    st.subheader("📡 Signal Strength (SNR) Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["snr"], bins=40, color="lightgreen", edgecolor="black")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Signal Strength (SNR)")
    st.pyplot(fig)
    st.markdown("""
    **Insight:**  
    Higher Signal Strength (SNR) means the owl was closer to the tower.  
    Lower Signal Strength (SNR) may indicate movement away from the detection area.
    """)

    # ---------------------------
    # 3. Noise distribution
    # ---------------------------
    st.subheader("🌫 Noise Level Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["noise"], bins=40, color="salmon", edgecolor="black")
    ax.set_xlabel("Noise")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Noise Levels")
    st.pyplot(fig)
    st.markdown("""
    **Insight:**  
    Rising noise levels can precede declines in SNR and may signal early movement or environmental changes.
    """)

    # ---------------------------
    # 4. Movement class distribution 
    # ---------------------------
    if "movement" in df.columns:
        st.subheader("🦉 Movement vs Resident — Class Balance")
        fig, ax = plt.subplots(figsize=(6, 4))
        df["movement"].value_counts().plot(kind="bar", color=["orange", "blue"], ax=ax)
        ax.set_xticklabels(["Resident (0)", "Movement (1)"], rotation=0)
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Movement Labels")
        st.pyplot(fig)

        # Add percentage breakdown
        movement_ratio = df["movement"].mean() * 100
        resident_ratio = 100 - movement_ratio

        st.markdown(f"""
        **Insight:**  
        - Resident (0): **{resident_ratio:.2f}%**  
        - Movement (1): **{movement_ratio:.2f}%**  

        Movement events are rare and occur in short windows, which is expected in telemetry data.
        """)

    # -------------------------------------------
    # ⭐ Movement Probability Over Time (Per Owl)
    # -------------------------------------------
    st.subheader("📈 Movement Probability Over Time")

    # Fix: convert datetime column to a real timedelta
    if "datetime" in df.columns:
        try:
            df["datetime"] = pd.to_timedelta(df["datetime"], errors="coerce")
        except:
            pass

        # Select owl through unique motusTagID
        if "motusTagID" in df.columns:
            owl_ids = df["motusTagID"].unique()
            selected_owl = st.selectbox("Choose an Owl (motusTagID)", owl_ids)
            owl_df = df[df["motusTagID"] == selected_owl].sort_values("datetime")
        else:
            owl_df = df.sort_values("datetime")

        # Compute movement probabilities
        X = owl_df[FEATURES].values
        movement_probs = clf.predict_proba(X)[:, 1]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(owl_df["datetime"], movement_probs, color="darkred")
        ax.axhline(0.30, linestyle="--", color="gray", label="Movement threshold (0.30)")
        ax.set_ylabel("Predicted Movement Probability")
        ax.set_xlabel("Time")
        ax.set_title("Movement Probability Trend Over Time")
        ax.legend()
        st.pyplot(fig)

        st.markdown("""
        **Insight:**  
        This chart shows how movement probability changes across time for a single owl.
        - Stable low values suggest local activity.  
        - Rising or unstable probabilities may reflect early signs of departure.  
        - Fluctuations often correspond to changes in SNR, noise, or lag features.
        """)
    else:
        st.info("Datetime column not found — unable to plot movement probability over time.")




# =============================================
# PAGE 3 — PREDICTION EXPLORER
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
# PAGE 4 — RAG EXPLANATION
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
