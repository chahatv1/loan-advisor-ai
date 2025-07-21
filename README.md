# Loan Advisor AI

**Loan Advisor AI** is an interactive chatbot that lets you analyze and explore loan approval data using natural language. It leverages Retrieval-Augmented Generation (RAG) to find real examples from a sample loan dataset, discover trends, and provide explainable, data-driven answers with charts and filters. Built with Streamlit and a local open-source language model.

---

## Features

- **Conversational analysis:** Ask questions about loan approvals and get instant, data-backed answers.
- **Retrieval-Augmented Generation (RAG):** Combines smart data search with AI-powered insights and summaries.
- **Interactive filters:** Explore trends by gender, credit history, property area, and more.
- **Automated charts:** Visualize key statistics and approval breakdowns.

---

## Tech Stack & Libraries

- **Python 3:** Main programming language for all components.
- **Streamlit:** User-friendly app framework for the interactive web UI.
- **pandas:** For data cleaning, manipulation, and applying filters to the loan dataset.
- **plotly:** To build dynamic and interactive charts in the dashboard.
- **faiss:** Efficient similarity search for finding the most relevant data rows based on user questions.
- **sentence-transformers:** Creates high-quality text embeddings from each data row and user queries, powering the retrieval part of RAG.
- **HuggingFace Transformers:** Loads and runs the language model to generate answers.
    - **Model used:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
    - **Why:** This model is fast, open-source, easily runnable on local machines, and requires no API keys or cloud access.
- **torch, accelerate:** Enhance speed and device compatibility for local LLM inference.
  
---

## Dataset

Sample data from [Loan Approval Prediction (Kaggle)](https://www.kaggle.com/datasets)  
Includes: gender, marital status, dependents, education, employment, applicant income, co-applicant income, loan amount, credit history, property area, and loan status columns.

---

## Author

**Created by Chahat Verma**  
Open-source for learning and demonstration purposes.
