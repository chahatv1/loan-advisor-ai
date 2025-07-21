import streamlit as st
import pandas as pd
from modules.preprocessing import load_and_clean_data, row_to_text
from modules.embedder import load_faiss_index
from modules.retriever import retrieve_similar_rows
from modules.generator import generate_explanation
from sentence_transformers import SentenceTransformer
import plotly.express as px

# ---- Project Paths ----
DATA_PATH = "data/loan_approval.csv"
INDEX_PATH = "data/loan_index.faiss"
METADATA_PATH = "data/loan_metadata.pkl"

# ---- Modern Font & UI Styling ----
st.set_page_config(page_title="Loan Advisor AI", layout="wide")
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', 'Lato', 'Montserrat', sans-serif;
        font-size: 17px;
        background-color: #f7fafc;
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        background: #fff;
        color: #222;
        font-size: 16px;
    }
    .stButton>button {
        font-size: 16px;
        border-radius: 6px;
        background-color: #176AD3;
        color: white;
        padding: 0.45rem 0.8rem;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Sidebar: Minimal, Professional ----
with st.sidebar:
    st.title("Loan Advisor AI")
    st.markdown(
        "**Powered by Retrieval-Augmented Generation (RAG):** Combines real-time search over actual loan records with state-of-the-art AI for explainable answers and interactive data visualizations."
    )
    st.markdown("---")
    st.markdown("*Dataset: Kaggle – Loan Approval Prediction*")

# ---- Data, Model, Index Cache ----
@st.cache_resource
def get_data_and_models():
    df = load_and_clean_data(DATA_PATH)
    faiss_index, metadata = load_faiss_index(INDEX_PATH, METADATA_PATH)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, faiss_index, metadata, model

df, faiss_index, metadata, model = get_data_and_models()

# ---- Title & Short Description ----
st.title("Loan Advisor AI")
st.markdown("Analyze loan approval decisions interactively—ask anything, see key trends, and get answers backed by real data and AI.")

# ---- Filters ----
with st.expander("🔎 Filter the dataset before you ask (optional):"):
    col1, col2, col3 = st.columns(3)
    with col1:
        gender_filter = st.selectbox("Gender", options=["All"] + sorted(df["gender"].dropna().unique()))
    with col2:
        credit_options = sorted(set([str(x) for x in df["credit_history"].dropna().unique()]))
        credit_filter = st.selectbox("Credit History", options=["All"] + credit_options)
    with col3:
        area_filter = st.selectbox("Property Area", options=["All"] + sorted(df["property_area"].dropna().unique()))

user_query = st.text_input("Ask your question about loan approvals:")

filtered_df = df.copy()
if gender_filter != "All":
    filtered_df = filtered_df[filtered_df["gender"] == gender_filter]
if credit_filter != "All":
    try:
        filtered_df = filtered_df[filtered_df["credit_history"] == float(credit_filter)]
    except:
        filtered_df = filtered_df[filtered_df["credit_history"] == credit_filter]
if area_filter != "All":
    filtered_df = filtered_df[filtered_df["property_area"] == area_filter]

if st.button("Ask"):
    if filtered_df.empty:
        st.warning("⚠️ No data matches your filters. Adjust them or reset.")
    elif not user_query.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("Analyzing your question..."):
            row_texts = filtered_df.apply(row_to_text, axis=1).tolist()
            rows, indices = retrieve_similar_rows(user_query, model, faiss_index, metadata)
            context = "\n".join(rows)
            relevant_df = df.iloc[list(indices)]
            openai_key = st.secrets["OPENAI_API_KEY"]
            answer = generate_explanation(context, user_query, openai_key)

        st.subheader("🧠 AI-Powered Analyst Answer")
        st.write(answer)

        st.subheader("📊 Visual Insight")
        try:
            fig = px.histogram(
                filtered_df,
                x="gender",
                color="loan_status",
                barmode="group",
                title="Loan Approval Status by Gender"
            )
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("No chart available for this query.")

        st.subheader("📄 Relevant Data Rows")
        st.dataframe(relevant_df, use_container_width=True)

# ---- Author Attribution / Footer ----
st.markdown(
    f"""<div style='text-align: right; color: #7a7a7a; font-size: 15px; padding: 1em 0 0.5em 0;'>
    Created by Chahat Verma
    </div>
    """,
    unsafe_allow_html=True
)
