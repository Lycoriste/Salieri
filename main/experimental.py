import torch
import torch.nn as nn
import math, random

"""

  Here lies all experimental and WIP networks

"""

class Sync2Bind(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 n_heads: int = 4, 
                 head_hidden_dim: int = 32, 
                 output_dim: int = 1, 
                 freq: float = 1000.0
        ):
        super().__init__()
        self.n_heads = n_heads
        self.freq = freq
        
        self.shared_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.heads = nn.ModuleList([
            nn.LSTM(hidden_dim, head_hidden_dim, batch_first=True) for _ in range(n_heads)
        ])
        
        self.readout = nn.Sequential(
            nn.Linear(n_heads * head_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, t):
        # x: (batch, seq_len, input_dim)
        shared_out, _ = self.shared_lstm(x)
        
        head_outputs = []
        for head in self.heads:
            head_out, _ = head(shared_out)
            mod_signal = torch.sin(2 * math.pi * self.freq * t).unsqueeze(-1)

            head_out = head_out * mod_signal
            head_outputs.append(head_out[:, -1, :])
        
        combined = torch.cat(head_outputs, dim=-1)
        out = self.readout(combined)
        return out

class AdaptiveCorippleLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads=2, init_prob=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.heads = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(n_heads)])
        
        # Learnable burst logits, dynamic rippling probability
        self.burst_logits = nn.Parameter(torch.full((n_heads,), torch.logit(torch.tensor(init_prob))))
    
    def forward(self, x):
        print(x.shape)
        batch, state_dim = x.shape
        outputs = []

        # Compute burst probabilities from logits
        burst_probs = torch.sigmoid(self.burst_logits)  # (n_heads,)
        
        for i, head in enumerate(self.heads):
            h = head(x)
            burst_mask = (torch.rand(batch, self.hidden_dim, device=x.device) < burst_probs[i]).float()
            h = h * burst_mask
            outputs.append(h)

        out = torch.cat(outputs, dim=-1)
        return out