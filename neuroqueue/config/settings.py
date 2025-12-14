import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Mongo
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB = os.getenv("MONGO_DB", "neuroqueue")

    # ChromaDB
    CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
    CHROMA_TENANT = os.getenv("CHROMA_TENANT", "default_tenant")
    CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "default_database")

    # AI
    MODEL_DIR = os.getenv("MODEL_DIR", "models")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

settings = Settings()
