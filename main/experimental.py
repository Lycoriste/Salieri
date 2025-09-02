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
        self.out_dim = hidden_dim * n_heads

        self.projection = nn.Linear(input_dim, self.out_dim)
        nn.init.xavier_uniform_(self.projection.weight, gain=1.0)
        nn.init.zeros_(self.projection.bias)
        
        # Learnable burst logits, dynamic rippling probability
        eps = 1e-6
        init_prob = torch.tensor(init_prob).clamp(eps, 1 - eps)
        self.burst_logits = nn.Parameter(torch.full((n_heads,), torch.logit(init_prob)))

        self.input_norm = nn.LayerNorm(input_dim, eps=eps)
        self.output_norm = nn.LayerNorm(self.out_dim, eps=eps)
    
    def forward(self, x):
        x = self.input_norm(x)
        h = self.projection(x)
        h = h.view(-1, self.n_heads, self.hidden_dim)

        # Compute burst probabilities from logits
        if self.training:
            burst_probs = torch.sigmoid(self.burst_logits).view(1, self.n_heads, 1).clamp(1e-3, 0.3)
            burst_mask = (torch.rand_like(h) < burst_probs).float()
            h = h * burst_mask
            h = torch.tanh(h)

        out = h.view(-1, self.out_dim)
        out = self.output_norm(out)
        return out