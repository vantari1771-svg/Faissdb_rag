# rag_utility.py  (FAISS version)

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# 0. ENV + GLOBALS
# -----------------------------

load_dotenv()

# Folder where this file lives
working_dir = os.path.dirname(os.path.abspath(__file__))

# Where to store the FAISS index on disk
INDEX_DIR = os.path.join(working_dir, "faiss_index")

# Embeddings model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

# Prompt template + parser
prompt = ChatPromptTemplate.from_template(
    """
Use ONLY the context below to answer the question.
If the answer is not in the context, say: "Not in provided context."

Context:
{context}

Question:
{question}
"""
)

parser = StrOutputParser()


# -----------------------------
# 1. PDF -> FAISS index
# -----------------------------

def process_document_to_faiss_db(file_name: str) -> None:
    """
    1. Load a PDF from working_dir/file_name
    2. Split into chunks
    3. Create a FAISS vector index and save it to INDEX_DIR
    (overwrite each time you upload a new doc)
    """
    pdf_path = os.path.join(working_dir, file_name)

    # 1) Load PDF
    loader = UnstructuredPDFLoader(pdf_path)
    documents = loader.load()

    # 2) Split into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)

    # 3) Build FAISS index in memory
    vectordb = FAISS.from_documents(chunks, embedding)

    # 4) Save FAISS index to disk
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectordb.save_local(INDEX_DIR)


# -----------------------------
# 2. Question -> Answer
# -----------------------------

def answer_question(user_question: str) -> str:
    """
    1. Load FAISS index from INDEX_DIR
    2. Retrieve top-k chunks for the question
    3. Ask Groq LLM using those chunks as context
    """
    # 1) Load FAISS index from disk
    vectordb = FAISS.load_local(
        INDEX_DIR,
        embedding,
        allow_dangerous_deserialization=True,  # required by LangChain
    )

    # 2) Build retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # 3) Get relevant docs
    docs = retriever.invoke(user_question)
    context = "\n\n".join(d.page_content for d in docs)

    # 4) Run LLM chain
    chain = prompt | llm | parser
    answer = chain.invoke({"context": context, "question": user_question})

    return answer
