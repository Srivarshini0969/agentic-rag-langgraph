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
import re

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# -------------------------------------------
# Load environment variables
# -------------------------------------------
# Reads environment variables from the .env file.
# We need GOOGLE_API_KEY to generate embeddings using Gemini.

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY is missing in .env")


# -------------------------------------------
# File paths
# -------------------------------------------
# DATA_FILE: Contains our Rogers FAQ questions and answers.
# INDEX_PATH: Location where the generated FAISS index is saved.

DATA_FILE = "data/info.txt"
INDEX_PATH = "faiss_index"


# -------------------------------------------
# Build vectorstore
# -------------------------------------------

def build_vectorstore():

    print("🔄 Rebuilding FAISS vectorstore...")

    # -------------------------------------------
    # Load content
    # -------------------------------------------
    # Check whether the source FAQ file exists.
    # Read the complete file as a single string.

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"❌ Cannot find {DATA_FILE}"
        )

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # Stop if the source file is empty.
    if not text.strip():
        raise ValueError(
            "❌ data/info.txt is empty."
        )

    # -------------------------------------------
    # Split text into FAQ documents
    # -------------------------------------------
    # OLD APPROACH:
    # RecursiveCharacterTextSplitter divided the entire document
    # into chunks of 1000 characters with 200 characters of overlap.
    #
    # PROBLEM:
    # A chunk could contain parts of multiple FAQs or split an
    # answer across chunks.
    #
    # NEW APPROACH:
    # Split the document immediately before each heading such as:
    # ### Q1:, ### Q2:, ### Q3:, etc.
    #
    # This keeps each question and its answer together.
    # (?=...) is a regex lookahead: it finds the heading without
    # removing that heading from the resulting section.

    faq_sections = re.split(
        r"(?=### Q\d+:)",
        text
    )

    # Store the complete FAQ sections as LangChain Documents.
    docs = []

    for section in faq_sections:

        # Remove unnecessary whitespace at the beginning and end.
        section = section.strip()

        # Keep only sections that begin with a numbered FAQ heading.
        # This excludes the document title and page headings.
        if re.match(r"### Q\d+:", section):

            docs.append(
                Document(page_content=section)
            )

    print(f"📄 Created {len(docs)} FAQ documents.")

    # Stop if no FAQ sections were found.
    if not docs:
        raise ValueError(
            "❌ No FAQ sections found in data/info.txt"
        )

    # -------------------------------------------
    # Gemini Embeddings
    # -------------------------------------------
    # Convert each FAQ document into a numerical vector.
    # These vectors allow FAISS to find FAQs semantically
    # related to a user's question.
    #
    # We retain the same embedding model and dimensionality
    # as the existing implementation.

    embedder = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=768,
        google_api_key=API_KEY,
    )

    # -------------------------------------------
    # Build FAISS vectorstore
    # -------------------------------------------
    # Create a searchable vector index from the FAQ documents
    # and their embeddings.

    faiss_db = FAISS.from_documents(
        docs,
        embedder
    )

    # -------------------------------------------
    # Save vectorstore
    # -------------------------------------------
    # Save the generated index so langgraph_runner.py can load
    # it when the application starts.

    faiss_db.save_local(INDEX_PATH)

    print(f"✅ Vectorstore saved to ./{INDEX_PATH}/")

    print("\n🎉 Completed successfully!")
    print("You can now run:")
    print("    python app.py")


# -------------------------------------------
# Run script
# -------------------------------------------
# Execute build_vectorstore() only when this file is run directly.
# Importing this file into another Python module will not
# automatically rebuild the index.

if __name__ == "__main__":
    build_vectorstore()