import torch
import torch.nn as nn
import torch.nn.functional as F

class IntentClassifier(nn.Module):
    def __init__(self, input_dim=384, num_classes=6):
        super(IntentClassifier, self).__init__()
        # Simple MLP: Input -> Hidden -> Output
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.classes = ["payment", "notification", "alert", "logs", "analytics", "critical"]

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Raw logits
        return x

    def predict(self, embedding: torch.Tensor):
        self.eval()
        with torch.no_grad():
            # Ensure batch dimension
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            
            logits = self.forward(embedding)
            prob = F.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(prob, 1)
            
            return self.classes[predicted_idx.item()], confidence.item()
