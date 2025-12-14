from sentence_transformers import SentenceTransformer
import torch
from neuroqueue.config.logger import logger

class MessageEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise e

    def embed(self, text: str) -> torch.Tensor:
        """Generates a tensor embedding for the text."""
        embedding = self.model.encode(text, convert_to_tensor=True)
        return embedding

# Global instance
embedder = MessageEmbedder()
