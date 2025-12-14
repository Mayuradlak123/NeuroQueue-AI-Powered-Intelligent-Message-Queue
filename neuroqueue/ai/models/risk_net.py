import torch
import torch.nn as nn
from neuroqueue.ai.layers.attention import SelfAttention

class RiskTransformer(nn.Module):
    """
    Predicts system failure risk using a sequence of metrics processed by a Transformer.
    """
    def __init__(self, input_dim=4, seq_len=5, heads=1):
        super(RiskTransformer, self).__init__()
        
        self.embed_dim = 16 # Small dimension for metrics
        self.input_proj = nn.Linear(input_dim, self.embed_dim)
        
        # Positional Encoding (Simple learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, self.embed_dim))
        
        # Custom Attention Block (Temporal Attention)
        self.attention = SelfAttention(self.embed_dim, heads)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        
        # Feed Forward
        self.ff = nn.Sequential(
            nn.Linear(self.embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.embed_dim)
        )
        self.norm2 = nn.LayerNorm(self.embed_dim)
        
        # Output Head
        # Flatten (seq_len * embed) -> 1
        self.head = nn.Linear(seq_len * self.embed_dim, 1)

    def forward(self, x):
        # x: (Batch, Seq, InputDim)
        
        # 1. Project & Add Positional Info
        x = self.input_proj(x) + self.pos_embedding
        
        # 2. Self Attention
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # 3. Feed Forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        # 4. Flatten & Predict
        N = x.shape[0]
        x = x.reshape(N, -1)
        
        risk = torch.sigmoid(self.head(x))
        return risk

    def predict(self, seq_data: torch.Tensor):
        self.eval()
        with torch.no_grad():
            if seq_data.dim() == 2:
                seq_data = seq_data.unsqueeze(0)
            return self.forward(seq_data).item()
