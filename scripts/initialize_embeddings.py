from app.rag.vector_store import create_vector_store

# Load your documents here (example with a small corpus)
with open("data/docs/docs.txt", "r") as f:
    documents = f.read()

# Create the FAISS vector store
vector_store = create_vector_store(documents)

# Optionally, save the vector store to a file
vector_store.save_local("faiss_index")
print("Embeddings precomputed and saved!")
