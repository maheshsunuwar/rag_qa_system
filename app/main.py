import time
import streamlit as st
from rag.qa_pipeline import generate_answer
from rag.vector_store import create_vector_store
from rag.retriever import retrieve_docs

import fitz # pymupdf
st.set_page_config(page_title="Chat with PDF")

# @traceable
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page in doc:
        text+=page.get_text()

    return text

# @traceable
def evaluate_query(query, expected_answer, vector_store, llm_model):
    start_time = time.time()

    docs = retrieve_docs(vector_store, query)
    answer = generate_answer(docs, query, llm_model)

    latency = time.time() - start_time

    # Evaluate response quality (this can be enhanced with more sophisticated methods)
    quality = "High" if expected_answer in answer else "Low"

    return {
        'answer': answer,
        'latency': latency,
        'quality': quality
    }

# with open("data/docs/docs.txt", "r") as f:
#         documents = f.read()
# vector_store = create_vector_store(documents=documents)
# st.write(vector_store)
st.write("Created new FAISS vector store.")


# st.title("RAG QA")

# started with just a simple query
# query = st.text_input('Ask a question')

# if query:
#     with st.spinner("Generating Answer..."):
#         docs = retrieve_docs(vector_store=vector_store, query=query)
#         answer = generate_answer(docs, question=query)
#         st.subheader("Answer:")
#         st.write(answer)

#         st.subheader("Retrieved Context:")
#         for doc in docs:
#             st.write("- ", doc)

# updating the code for pdf
# st.title("RAG QA with PDF")

def app():
    uploaded_pdf = st.file_uploader("Upload a PDF document", type="pdf")
    if uploaded_pdf:
        with st.spinner('Processing PDF...'):
            vector_store = create_vector_store(uploaded_pdf=uploaded_pdf)
            st.write(f"PDF Loaded.")
            query = st.text_input('Ask a question:')

            if query:
                with st.spinner('Generating Answer...'):
                    docs = retrieve_docs(vector_store=vector_store, query=query)
                    answer = generate_answer(docs, query)
                    st.subheader('Answer:')
                    st.write(answer)
                    st.subheader('Retrieved Context:')
                    for doc in docs:
                        st.write('- ', docs)
