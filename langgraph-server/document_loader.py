# document_loader.py

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(file_path: str = "data/info.txt"):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)
