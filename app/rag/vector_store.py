import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader
import tempfile

from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# from langchain.embeddings import Embeddings
import faiss
import config
import pickle

def load_pdf(uploaded_pdf):
    # loader = PyMuPDFLoader(pdf_path)
    # documents = loader.load()

    # Create a temporary file to store the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(uploaded_pdf.getvalue())
        tmpfile_path = tmpfile.name

    # Load the PDF using LangChain's PyPDFLoader
    # loader = PyPDFLoader(tmpfile_path)
    loader = PyMuPDFLoader(tmpfile_path, mode='single')
    documents = loader.load()
    print(documents[0].page_content)
    return documents[0].page_content


def create_vector_store(documents=None, uploaded_pdf= None, save_path="faiss_index", ):
    ollama_embeddings = OllamaEmbeddings(
        base_url=config.OLLAMA_URL,
        model=config.OLLAMA_MODEL,
    )
    if uploaded_pdf:
        documents = load_pdf(uploaded_pdf)

    # text_splitter = CharacterTextSplitter()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", r"(?<=[.?!])\s+"],  # Define separators, including regex
        chunk_size=1000,  # Maximum chunk size
        chunk_overlap=100,  # Overlap between chunks
        is_separator_regex=True  # Treat separators as regular expressions
    )

    texts = text_splitter.split_text(documents)

    vector_store = FAISS.from_texts(texts=texts, embedding=ollama_embeddings)

    return vector_store
