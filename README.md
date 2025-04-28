## Retrieval-Augmented Generation (RAG) QA System
Welcome to my project! This is a lightweight Retrieval-Augmented Generation (RAG) system where users can ask questions, and the system retrieves the most relevant documents and generates accurate answers using a language model (LLM).

## Features
- Build a vector database (FAISS) from document corpus
- Retrieve top-k relevant contexts based on semantic similarity
- Answer user questions grounded on retrieved documents
- Streamlit Frontend for interactive querying
- OpenAI Embeddings + GPT-3.5 Turbo for best results
- Modular design for easy extensions and improvements

## Technologies Used
- Python 3.10+
- Streamlit
- FAISS
- LangChain
- Ollama/openai


## Project Structure
```
rag_qa_system/
├── README.md
├── data/
│   └── docs/
│       └── sample_docs.txt
├── app/
│   ├── main.py
│   ├── rag/
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   └── qa_pipeline.py
│   └── config.py
├── requirements.txt
└── scripts/
    └── initialize_embeddings.py
```

## Getting Started
### Clone the repository:
```
git clone https://github.com/yourusername/rag_qa_system.git
cd rag_qa_system
```

### Install dependencies:
```
pip install -r requirements.txt
```

### Set your OpenAI API key:
#### Create a .env file:
```
If you use openai instead of ollama
OPENAI_API_KEY=your_openai_api_key_here
```

### Launch the app:
```
streamlit run app/main.py
```

## How It Works
- Ingest Documents → Split into smaller chunks
- Embed Documents → Using OpenAI Embeddings
- Store Embeddings → In FAISS vector database
- Retrieve Relevant Chunks → Based on user question
- Generate Answer → Using GPT-3.5 Turbo based on retrieved context

## Future Enhancements
- Upload and ingest PDFs and Docs
- Fine-tune retrieval for domain-specific datasets
- Add evaluation metrics (e.g., retrieval precision, latency)
- Deploy live on HuggingFace Spaces or AWS

🙋‍♂️ About Me
I'm Mahesh Sunuwar — a passionate machine learning engineer specializing in  RAG systems, LLMs, and MLOps!
