import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """
    Explicit implementation of Scaled Dot-Product Attention.
    Demonstrates understanding of Q, K, V mechanics.
    """
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # Linear projections for Query, Key, Value
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        # Output projection
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        # inputs are (N, seq_length, embed_size)
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Projections
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # MatMul: Q * K^T
        # shapes: (N, heads, query_len, head_dim) * (N, heads, head_dim, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Scale
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        energy = energy / math.sqrt(self.head_dim)

        if mask is not None:
             energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Softmax
        attention = torch.softmax(energy, dim=3) # (N, heads, query_len, key_len)

        # MatMul: Attention * V
        out = torch.einsum("nhqk,nvhd->nqhd", [attention, values])
        
        # Flatten heads
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        # Final Linear layer
        out = self.fc_out(out)
        return out
