# app.py

import os
import streamlit as st

from rag_utility import process_document_to_faiss_db, answer_question

# Folder where app.py lives
working_dir = os.path.dirname(os.path.abspath(__file__))

st.title("📄 Vin RAG – FAISS + Groq PDF Q&A")

st.write("Upload a PDF, I’ll index it with FAISS, then you can ask questions about it.")

# 1) Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file to the project directory
    save_path = os.path.join(working_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info(f"✅ Saved `{uploaded_file.name}` to project folder.")

    # Build FAISS index from this PDF
    process_document_to_faiss_db(uploaded_file.name)
    st.success("🚀 Document processed and FAISS index created.")

# 2) Question box
user_question = st.text_area("Ask your question about the uploaded document:")

# 3) Ask button
if st.button("💬 Get Answer"):
    if not user_question.strip():
        st.warning("Type a question first.")
    else:
        try:
            answer = answer_question(user_question)
            st.markdown("### 🧠 Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error while answering: {e}")
