import streamlit as st
from rag.qa_pipeline import generate_answer
from rag.vector_store import create_vector_store
from rag.retriever import retrieve_docs

with open("data/docs/docs.txt", "r") as f:
        documents = f.read()
vector_store = create_vector_store(documents=documents)
# st.write(vector_store)
st.write("Created new FAISS vector store.")


st.title("RAG QA")

query = st.text_input('Ask a question')

if query:
    with st.spinner("Generating Answer..."):
        docs = retrieve_docs(vector_store=vector_store, query=query)
        answer = generate_answer(docs, question=query)
        st.subheader("Answer:")
        st.write(answer)

        st.subheader("Retrieved Context:")
        for doc in docs:
            st.write("- ", doc)
