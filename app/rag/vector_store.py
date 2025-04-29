import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# from langchain.embeddings import Embeddings
import faiss
import config
import pickle

def create_vector_store(documents, save_path="faiss_index"):
    ollama_embeddings = OllamaEmbeddings(
        base_url=config.OLLAMA_URL,
        model=config.OLLAMA_MODEL,
    )
    text_splitter = CharacterTextSplitter()

    texts = text_splitter.split_text(documents)

    vector_store = FAISS.from_texts(texts=texts, embedding=ollama_embeddings)

    return vector_store
