import torch
from neuroqueue.ai.embeddings import embedder
from neuroqueue.features.text_stats import FeatureEngineer

class NeuroETL:
    """
    Extract, Transform, Load pipeline for NeuroQueue Message processing.
    """
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
    
    def process_message(self, content: str):
        # 1. Extract & Embed
        # E: Raw Text -> Vector
        # output shape: (384,)
        embedding = embedder.embed(content)
        
        # 2. Feature Engineering
        # T: Raw Text -> Stats Vector
        stats_list = self.feature_engineer.extract_features(content)
        stats_tensor = torch.tensor(stats_list, dtype=torch.float32)
        
        # 3. Load & Concatenate
        # L: Combine into model input vector
        # output shape: (394,)
        combined_features = torch.cat((embedding, stats_tensor), dim=0)
        
        return combined_features

etl_pipeline = NeuroETL()
