import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    MODEL_NAME = 'gemini-2.0-flash'
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
