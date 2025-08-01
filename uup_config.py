import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    # MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    MODEL_NAME = 'gemini-2.0-flash'
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
