import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SOURCE_MONGODB_URI = os.getenv("SOURCE_MONGODB_URI", "mongodb+srv://shubham:ahRwugDfrVr1eA16@nuvo.zt3sn.mongodb.net/?retryWrites=true&w=majority&appName=nuvo")
    SOURCE_DB_NAME = os.getenv("SOURCE_DB_NAME", "trxnmodb9u24cohskemabswz")
    SOURCE_COLLECTION_NAME = os.getenv("SOURCE_COLLECTION_NAME", "trxnmovs24ns9vc")
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    # MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    MODEL_NAME = 'gemini-2.0-flash'
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
