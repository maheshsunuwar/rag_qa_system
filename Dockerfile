# use lightweight python
FROM python:3.9-slim

# set working dir
WORKDIR ./

# copy code
COPY requirements.txt requirements.txt
COPY . .

# install system dependencies
# RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN pip install --upgrade pip
RUN pip install uv
RUN uv pip install -r requirements.txt --system
RUN uv pip install streamlit --system
RUN uv pip install pydantic --system
RUN uv pip install dotenv --system
RUN uv pip install psycopg2-binary --system
RUN uv pip install fitz --system
RUN uv pip install langchain --system
RUN uv pip install langchain-community --system
RUN uv pip install langchain-ollama --system
RUN uv pip install ollama --system
RUN uv pip install faiss-cpu --system
RUN uv pip install tiktoken --system
RUN uv pip install python-dotenv --system
RUN uv pip install pypdf --system
RUN uv pip install fuzzywuzzy --system
RUN uv pip install langsmith --system
RUN uv pip install pypdf --system

# expose port
EXPOSE 8999

# # run fastapi server
# CMD ["streamlit", "run", "app.py" "--server.address", "0.0.0.0", "--server.port", "8999"]
