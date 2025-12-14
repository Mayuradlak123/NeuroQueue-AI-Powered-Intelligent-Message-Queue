import torch
import torch.nn as nn
import torch.nn.functional as F
from neuroqueue.ai.layers.attention import SelfAttention

class IntentAttentionNet(nn.Module):
    """
    Advanced Intent Classifier using Self-Attention.
    """
    def __init__(self, feature_dim=394, num_classes=6, heads=2):
        super(IntentAttentionNet, self).__init__()
        
        # We project features to a higher dim valid for multi-head attention
        self.embed_dim = 256
        self.projection = nn.Linear(feature_dim, self.embed_dim)
        
        # Custom Attention Layer
        self.attention = SelfAttention(self.embed_dim, heads)
        
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.fc = nn.Linear(self.embed_dim, num_classes)
        self.classes = ["payment", "notification", "alert", "logs", "analytics", "critical"]

    def forward(self, x):
        # x is (Batch, Features) -> Transform to (Batch, Seq=1, Features) for Attention
        x = x.unsqueeze(1) 
        
        x_proj = self.projection(x) # (Batch, 1, 256)
        
        # Attention block (Self-Attention: Q=K=V)
        attn_out = self.attention(x_proj, x_proj, x_proj)
        
        # Add & Norm (Residual Connection)
        x = self.norm(x_proj + attn_out)
        x = self.dropout(x)
        
        # Flatten
        x = x.mean(dim=1) 
        
        out = self.fc(x)
        return out

    def predict(self, features: torch.Tensor):
        self.eval()
        with torch.no_grad():
            if features.dim() == 1:
                features = features.unsqueeze(0)
            logits = self.forward(features)
            probs = F.softmax(logits, dim=1)
            conf, idx = torch.max(probs, 1)
            return self.classes[idx.item()], conf.item()
