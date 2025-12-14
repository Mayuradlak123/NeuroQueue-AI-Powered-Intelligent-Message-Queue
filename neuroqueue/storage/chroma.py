import chromadb
from chromadb.utils import embedding_functions
import os
from neuroqueue.config.logger import logger
from neuroqueue.config.settings import settings
from neuroqueue.ai.embeddings import embedder

# Adapter to make our system embedder compatible with Chroma
class SystemEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, input: list[str]) -> list[list[float]]:
        # embedder.embed return tensor, we need list of floats
        return [embedder.embed(text).tolist() for text in input]

class ChromaService:
    def __init__(self, collection_name="neuroqueue_metadata"):
        logger.info(f"Initializing ChromaService for collection: {collection_name}")
        
        api_key = settings.CHROMA_API_KEY
        tenant = settings.CHROMA_TENANT
        database = settings.CHROMA_DATABASE
        
        # USE SYSTEM EMBEDDER for alignment
        self.embedding_fn = SystemEmbeddingFunction()


        if api_key:
            # Cloud Client (User Preferred)
            try:
                logger.info("Connecting to ChromaDB Cloud...")
                self.client = chromadb.CloudClient(
                    api_key=api_key,
                    tenant=tenant,
                    database=database
                )
                # Use cosine similarity
                metadata = {"hnsw:space": "cosine"}
            except ValueError as e:
                # This catches the specific "Database ... does not match" error
                if "does not match" in str(e):
                    msg = (f"ChromaDB Configuration Error: The database name '{database}' "
                           f"does not match the one associated with yours API key. "
                           f"Please check your .env file or create the database in Chroma Cloud.")
                    logger.critical(msg)
                    raise ValueError(msg)
                raise e
            except Exception as e:
                logger.critical(f"Failed to connect to ChromaDB: {e}")
                raise e
        else:
            # Local Persistent Client (Fallback)
            logger.info("Connecting to local ChromaDB...")
            self.client = chromadb.PersistentClient(path="./chroma_db")
            metadata = None
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.embedding_fn,
            metadata=metadata
        )
        
        # Initialize second collection for logs/history as vectors
        self.history_collection = self.client.get_or_create_collection(
            name="neuroqueue_vector_history",
            embedding_function=self.embedding_fn,
            metadata=metadata
        )
        logger.info(f"ChromaDB collection '{collection_name}' and 'neuroqueue_vector_history' ready.")

    def store_metadata(self, doc_id, content, metadata):
        """
        Generic storage for metadata/contexts.
        """
        logger.info(f"Storing metadata for {doc_id}")
        self.collection.upsert(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def store_message_vector(self, message_id, content, analysis):
        """
        Stores processed message text + analysis in vector DB for semantic search.
        """
        text = f"Content: {content}\nAnalysis: {analysis}"
        metadata = {"message_id": message_id, "type": "processed_message"}
        
        logger.info(f"Storing message vector {message_id} in ChromaDB")
        self.history_collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[message_id]
        )

    def retrieve_similar_messages(self, query, n_results=5):
        """
        Retrieves relevant past messages.
        """
        results = self.history_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []

# Global instance
chroma_client = ChromaService()
