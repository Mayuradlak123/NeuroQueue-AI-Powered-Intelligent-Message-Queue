import torch
import os
from neuroqueue.ai.models.intent_net import IntentAttentionNet
from neuroqueue.ai.models.risk_net import RiskTransformer
from neuroqueue.ai.priority_model import PriorityPredictor 
from neuroqueue.config.logger import logger

# Note: PriorityPredictor can stay as MLP for now, or be upgraded similarly.

def train_advanced_models():
    os.makedirs("models", exist_ok=True)
    logger.info("Initializing and saving ADVANCED models...")

    # 1. Intent Model (Attention Based)
    # Input dim = 384 (BERT) + 10 (Stats) = 394
    intent_model = IntentAttentionNet(feature_dim=394) 
    torch.save(intent_model.state_dict(), "models/intent_net.pth")
    logger.info("Saved models/intent_net.pth")

    # 2. Risk Model (Transformer Based)
    risk_model = RiskTransformer()
    torch.save(risk_model.state_dict(), "models/risk_net.pth")
    logger.info("Saved models/risk_net.pth")
    
    # 3. Priority Model (Reused MLP for regression)
    priority_model = PriorityPredictor() # Accepts 384 only? Need to check if we want features here too.
    # Let's keep priority simple or update it. For now, we'll feed it just embeddings to save refactoring time.
    torch.save(priority_model.state_dict(), "models/priority_model.pth")
    logger.info("Saved models/priority_model.pth")

if __name__ == "__main__":
    train_advanced_models()
