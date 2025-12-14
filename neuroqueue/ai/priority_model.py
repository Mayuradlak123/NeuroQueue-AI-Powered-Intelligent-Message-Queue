import torch
import torch.nn as nn
import torch.nn.functional as F

class PriorityPredictor(nn.Module):
    def __init__(self, input_dim=384):
        super(PriorityPredictor, self).__init__()
        # Regression model
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1) # Single output for score 0.0-1.0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) # Sigmoid to bound 0-1
        return x

    def predict(self, embedding: torch.Tensor):
        self.eval()
        with torch.no_grad():
             if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
             score = self.forward(embedding)
             return score.item()
