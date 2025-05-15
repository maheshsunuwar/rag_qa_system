import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv('OLLAMA_URL')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')

# LANGSMITH_TRACING = os.getenv('LANGSMITH_TRACING')
# LANGSMITH_ENDPOINT = os.getenv('LANGSMITH_ENDPOINT')
# LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
