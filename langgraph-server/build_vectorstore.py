"""
build_vectorstore.py
--------------------

This script rebuilds the FAISS vectorstore used by the Agentic RAG system.

Run this script ONLY when:
- You update data/info.txt
- You add new documents
- The faiss_index folder is missing

Usage:
    python build_vectorstore.py
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------------------------
# Load environment variables
# -------------------------------------------

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY is missing in .env")


# -------------------------------------------
# File paths
# -------------------------------------------

DATA_FILE = "data/info.txt"
INDEX_PATH = "faiss_index"


# -------------------------------------------
# Build vectorstore
# -------------------------------------------

def build_vectorstore():

    print("🔄 Rebuilding FAISS vectorstore...")

    # Load content
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"❌ Cannot find {DATA_FILE}"
        )

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        raise ValueError(
            "❌ data/info.txt is empty."
        )

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = splitter.create_documents([text])

    print(f"📄 Split into {len(docs)} chunks.")


    # -------------------------------------------
    # Gemini Embeddings
    # -------------------------------------------

    embedder = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=768,
        google_api_key=API_KEY,
    )


    # -------------------------------------------
    # Build FAISS vectorstore
    # -------------------------------------------

    faiss_db = FAISS.from_documents(
        docs,
        embedder
    )


    # -------------------------------------------
    # Save vectorstore
    # -------------------------------------------

    faiss_db.save_local(INDEX_PATH)

    print(f"✅ Vectorstore saved to ./{INDEX_PATH}/")

    print("\n🎉 Completed successfully!")
    print("You can now run:")
    print("    python app.py")


# -------------------------------------------
# Run script
# -------------------------------------------

if __name__ == "__main__":
    build_vectorstore()