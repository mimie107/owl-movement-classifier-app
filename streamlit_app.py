import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import os

# =============================================
#  CLOUD-SAFE SETTINGS
# =============================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_TORCH"] = "1"

st.set_page_config(page_title="Owl Movement Classifier", page_icon="🦉", layout="wide")

st.title("🦉 Owl Movement Classifier — XGBoost + SHAP + RAG")
st.write("This app uses your **final cleaned classifier** with RAG + SHAP explanations.")


# =============================================
# 1 — LOAD XGBOOST CLASSIFIER (NEW MODEL)
# =============================================
@st.cache_resource
def load_model():
    return joblib.load("xgb_classifier.pkl")   # << USE YOUR ACTUAL MODEL

clf = load_model()


# =============================================
# 2 — NEW FEATURE LIST (ALIGNED WITH YOUR PIPELINE)
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
# 3 — LOAD USER DATASET
# =============================================
st.sidebar.header("📂 Upload Dataset")
uploaded = st.sidebar.file_uploader("Upload your processed df (with engineered features)", type=["csv"])

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

    # XGBoost binary returns shap_vals as array shape (n_samples, n_features)
    if isinstance(shap_vals, np.ndarray):
        return shap_vals[0]

    # Sometimes SHAP returns a list for binary models
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
# 5 — RAG EMBEDDINGS
# =============================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

rag_docs = {
    "movement_definition": """
Movement_class = 1 indicates a long gap in detections (time_diff > adaptive threshold),
suggesting temporary departure, vagrancy, or early migratory behavior.
""",
    "feature_explanation": """
Important predictors include signal strength (snr), noise trends, lag features,
and rolling averages which encode stability or disruption in detection patterns.
""",
    "model_info": """
This classifier is an XGBoost model using engineered features:
hour_sin, hour_cos, day, month, snr_lag1, snr_lag2, rolling noise/snr, and burstSlop.
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
# 6 — FLAN-T5 FOR NATURAL LANGUAGE EXPLANATION
# =============================================
@st.cache_resource
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-small", device="cpu")

generator = load_llm()

def llm_explain(row, shap_text, rag_text):
    prompt = f"""
You are an owl movement ecologist.

DATA ROW:
{row[FEATURES].to_string()}

SHAP SUMMARY:
{shap_text}

CONTEXT:
{rag_text}

Explain clearly why the model predicted movement vs residency.
"""
    return generator(prompt, max_new_tokens=200)[0]["generated_text"]


# =============================================
# SIDEBAR NAV
# =============================================
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Prediction Explorer", "🧠 RAG Explanation", "💬 Owl Chatbot"]
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
- FLAN-T5 reasoning  
""")


# =============================================
# PAGE 2 — PREDICTION EXPLORER
# =============================================
elif page == "🔍 Prediction Explorer":
    st.header("🔍 Prediction Explorer")

    idx = st.number_input("Row index:", min_value=0, max_value=len(df)-1, value=0)
    row = df.iloc[idx]

    X = row[FEATURES].values.reshape(1, -1)
    prob = clf.predict_proba(X)[0, 1]
    pred = int(prob >= 0.30)

    label = "Movement Event (1)" if pred == 1 else "Resident (0)"

    st.subheader("Prediction")
    st.write(f"**Predicted class:** {label}")
    st.write(f"**Probability of movement:** {prob:.3f}")

    # SHAP
    shap_vals = shap_for_row(row)
    st.subheader("SHAP Explanation")
    st.text(shap_text_summary(shap_vals))

    # Waterfall plot
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
    st.header("🧠 Model Explanation with RAG + LLM")

    idx = st.number_input("Row:", min_value=0, max_value=len(df)-1, value=0)
    row = df.iloc[idx]

    shap_vals = shap_for_row(row)
    shap_text = shap_text_summary(shap_vals)
    ctx = retrieve_context("movement prediction")

    explanation = llm_explain(row, shap_text, ctx)

    st.subheader("Explanation")
    st.write(explanation)


# =============================================
# PAGE 4 — CHATBOT
# =============================================
elif page == "💬 Owl Chatbot":
    st.header("💬 Owl Chatbot")

    q = st.text_input("Ask a question about owls or movement modeling:")

    if q:
        ctx = retrieve_context(q)
        prompt = f"Context:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer clearly."
        ans = generator(prompt, max_new_tokens=150)[0]["generated_text"]
        st.write(ans)
