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
#  CLOUD-SAFE SETTINGS (IMPORTANT)
# =============================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_TORCH"] = "1"

st.set_page_config(page_title="Owl Movement Classifier", page_icon="🦉", layout="wide")

st.title("🦉 Owl Movement Classifier — RAG + SHAP (Local) + LLM")
st.write("Fast, cloud-optimized version.")


# =============================================
# 1 — LOAD RANDOM FOREST MODEL
# =============================================
@st.cache_resource
def load_model():
    return joblib.load("owl_rf_binary.pkl")

rf_model = load_model()


# =============================================
# 2 — FEATURE LIST
# =============================================
feature_cols = [
    'snr','sig','sigsd','noise',
    'sig_lag1','sig_lag2',
    'snr_lag1','snr_lag2',
    'sig_roll3','sig_roll5',
    'time_diff','detections_cum',
    'sig_diff','noise_spike',
    'weak_signal','strong_signal'
]


# =============================================
# 3 — LOAD DATASET
# =============================================
st.sidebar.header("📂 Upload Dataset")
uploaded = st.sidebar.file_uploader("Upload df_fe_sample.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Dataset loaded!")
else:
    st.warning("Upload df_fe_sample.csv to continue.")
    st.stop()


# =============================================
# 4 — SHAP (LOCAL ONLY)
# =============================================
@st.cache_resource
def load_shap_explainer():
    return shap.TreeExplainer(rf_model)

explainer = load_shap_explainer()


def shap_for_row(row):
    X = row[feature_cols].values.reshape(1, -1)

    shap_out = explainer.shap_values(X)

    # Case 1 — SHAP returns [array(class0), array(class1)]
    if isinstance(shap_out, list) and len(shap_out) > 1:
        return shap_out[1][0]

    # Case 2 — SHAP returns only one array
    # (TreeExplainer with model_output="raw")
    if isinstance(shap_out, np.ndarray):
        return shap_out[0]

    # Fallback
    return shap_out[0][0]



def shap_text_summary(row, shap_vals):
    pairs = list(zip(feature_cols, shap_vals))
    top = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:5]

    text = "Top SHAP contributions:\n"
    for feat, val in top:
        direction = "increases migration likelihood" if val > 0 else "supports residency"
        text += f"- {feat}: {val:.3f} → {direction}\n"
    return text


# =============================================
# 5 — RAG EMBEDDINGS (CACHED)
# =============================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

rag_docs = {
    "model": """
    The owl movement classifier predicts Resident vs Migratory/Vagrant behavior.
    Long time_diff values indicate the owl may have left the receiver area.
    SHAP identifies which features affect each prediction.
    """,

    "features": """
    Important features include: time_diff, signal strength lags, SNR lags,
    rolling features, and cumulative detections.
    """
}

rag_embeddings = {
    k: embedder.encode(v, convert_to_tensor=True)
    for k, v in rag_docs.items()
}


def retrieve_context(query):
    q_emb = embedder.encode(query, convert_to_tensor=True)
    scores = {k: util.cos_sim(q_emb, emb).item() for k, emb in rag_embeddings.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return "\n\n".join([rag_docs[k] for k, _ in ranked[:2]])


# =============================================
# 6 — FLAN-T5 SMALL (FAST, CLOUD-SAFE)
# =============================================
@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device="cpu"
    )

generator = load_llm()


def llm_explain(row, shap_text, rag_text):
    prompt = f"""
You are an owl movement ecologist.

DATA ROW:
{row.to_string()}

SHAP SUMMARY:
{shap_text}

EXPERT CONTEXT:
{rag_text}

Explain movement behavior clearly.
"""

    out = generator(prompt, max_new_tokens=180, temperature=0.5)[0]["generated_text"]
    return out


# =============================================
# 7 — SIDEBAR NAVIGATION
# =============================================
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Prediction Explorer", "🧠 RAG Movement Explanation", "💬 Owl Chatbot", "📘 Documentation"]
)


# =============================================
# PAGE 1 — HOME
# =============================================
if page == "🏠 Home":
    st.header("Welcome")
    st.write("""
    Cloud-optimized owl movement classifier.

    ### Included:
    - RandomForest predictions  
    - SHAP local explainability  
    - RAG (retrieval-augmented context)  
    - FLAN-T5-Small natural language reasoning  
    - Multi-page Streamlit interface  
    """)



# =============================================
# PAGE 2 — Prediction Explorer
# =============================================
elif page == "🔍 Prediction Explorer":
    st.header("🔍 Prediction Explorer")

    idx = st.number_input("Select Row Index", min_value=0, max_value=len(df)-1, value=0)
    row = df.iloc[idx]

    X = row[feature_cols].values.reshape(1, -1)
    pred = rf_model.predict(X)[0]
    prob = rf_model.predict_proba(X).max()

    label = {0: "Resident", 1: "Migratory/Vagrant"}[pred]

    st.subheader("Prediction")
    st.write(f"**Movement Type:** {label}")
    st.write(f"**Confidence:** {prob:.2f}")

    # SHAP local explanation
    shap_vals = shap_for_row(row)
    st.subheader("SHAP Local Explanation")
    st.text(shap_text_summary(row, shap_vals))

    # Waterfall plot
    st.subheader("SHAP Waterfall Plot")
    expl = shap.Explanation(values=shap_vals, feature_names=feature_cols, features=X)
    fig = shap.plots.waterfall(expl, show=False)
    st.pyplot(fig)



# =============================================
# PAGE 3 — RAG + LLM Movement Explanation
# =============================================
elif page == "🧠 RAG Movement Explanation":
    st.header("🧠 RAG + LLM Movement Explanation")

    idx = st.number_input("Select Row", min_value=0, max_value=len(df)-1, value=0)
    row = df.iloc[idx]

    shap_vals = shap_for_row(row)
    shap_text = shap_text_summary(row, shap_vals)
    ctx = retrieve_context("owl movement")

    explanation = llm_explain(row, shap_text, ctx)

    st.subheader("Explanation")
    st.write(explanation)



# =============================================
# PAGE 4 — Owl Chatbot
# =============================================
elif page == "💬 Owl Chatbot":
    st.header("💬 Owl Movement Chatbot")

    query = st.text_input("Ask anything about owls, features, or predictions:")

    if query:
        ctx = retrieve_context(query)
        prompt = f"""
Background:
{ctx}

Question:
{query}

Answer clearly.
"""
        out = generator(prompt, max_new_tokens=150)[0]["generated_text"]
        st.write(out)



# =============================================
# PAGE 5 — Documentation
# =============================================
elif page == "📘 Documentation":
    st.header("📘 Documentation")
    st.write("""
    This system combines ML, SHAP, RAG, and LLMs to explain owl movement behavior.
    """)

