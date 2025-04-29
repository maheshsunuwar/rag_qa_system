from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import numpy as np
# from ollama import embed
import config

def retrieve_docs(vector_store, query, k = 3):
    """
    Retrieves the top-k most relevant documents based on the user's query.

    Args:
    - vector_store: The FAISS vector store containing the embeddings.
    - query: The user's query to find relevant documents.
    - k: The number of documents to retrieve (default is 3).

    Returns:
    - A list of the top-k most relevant documents.
    """

    # convert the query into embeddings using ollama
    ollama_embeddings = OllamaEmbeddings(
        base_url=config.OLLAMA_URL,
        model=config.OLLAMA_MODEL
    )
    # response = ollama_embeddings.embed_query(query)

    # perform similarity search on the FAISS index
    docs = vector_store.similarity_search(query, k)

    # retrieve the documents corresponding to the indices
    # docs = [vector_store.docs[i] for i in docs]

    return [doc.page_content for doc in docs]
