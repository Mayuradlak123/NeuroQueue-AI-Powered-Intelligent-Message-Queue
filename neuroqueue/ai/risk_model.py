import torch
import torch.nn as nn

class RiskAssessor(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, num_layers=1):
        super(RiskAssessor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # Probability of failure

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        # e.g. [msg_rate, consumer_lag, retries, latency] over last 5 ticks
        out, _ = self.lstm(x)
        # Take the last time step
        last_out = out[:, -1, :]
        risk = torch.sigmoid(self.fc(last_out))
        return risk
    
    def predict(self, sequence_data: torch.Tensor):
        # sequence_data shape: (seq_len, 4)
        self.eval()
        with torch.no_grad():
            if sequence_data.dim() == 2:
                sequence_data = sequence_data.unsqueeze(0)
            
            risk = self.forward(sequence_data)
            return risk.item()
