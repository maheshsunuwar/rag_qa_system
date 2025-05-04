
from langchain_ollama import ChatOllama

from app import config

def generate_answer(retrieved_docs, question):
    context = '\n\n'.join(retrieved_docs)

    prompt = f"""
    Answer the question based on the following context: \n{context}\n\n
    Question: {question}
    """

    # generate an answer using llama3.2:3b
    llm = ChatOllama(
        base_url=config.OLLAMA_URL,
        model='llama3.2:3b',
    )
    response = llm.invoke(prompt)
    # return response
    return response.content
