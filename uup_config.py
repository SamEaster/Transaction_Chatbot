import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SOURCE_MONGODB_URI = os.getenv("SOURCE_MONGODB_URI")
    SOURCE_DB_NAME = os.getenv("SOURCE_DB_NAME")
    SOURCE_COLLECTION_NAME = os.getenv("SOURCE_COLLECTION_NAME")
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    MODEL_NAME = 'gemini-2.0-flash'
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
