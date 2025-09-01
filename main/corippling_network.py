from experimental import AdaptiveCorippleLayer

import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution, TanhTransform, Bernoulli

# --- Experimental corippling networks --- #
class COR_Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hidden_dim = 32

        self.ripple_layer = AdaptiveCorippleLayer(
            state_dim,
            hidden_dim,
            n_heads=action_dim,
            init_prob=0.15
        )
        units = hidden_dim * action_dim

        self.move_head = nn.Linear(units, 1)
        self.turn_mu_head = nn.Linear(units, 1)
        self.turn_log_std_head = nn.Linear(units, 1)

        self.apply(self._init_weights)
        
        # Initialize turn_mu bias to 0 to prevent local minima
        nn.init.constant_(self.turn_mu_head.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = self.ripple_layer(state)
        
        move_logits = self.move_head(x)
        turn_mu = self.turn_mu_head(x)
        turn_log_std = torch.clamp(self.turn_log_std_head(x), -10, 2)
        
        return move_logits, turn_mu, turn_log_std

    def get_action_distribution(self, state):
        state_tensor = torch.tensor(state)
        move_logits, turn_mu, turn_log_std = self.forward(state_tensor)
        
        # Discrete movement distribution
        move_dist = Bernoulli(logits=move_logits.squeeze(-1))
        
        # Continuous turning distribution with tanh squashing
        turn_std = torch.exp(turn_log_std)
        normal_dist = Normal(turn_mu.squeeze(-1), turn_std.squeeze(-1))
        turn_dist = TransformedDistribution(normal_dist, TanhTransform())
        
        return move_dist, turn_dist


class COR_Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hidden_dim = 32

        self.ripple_layer = AdaptiveCorippleLayer(
            state_dim + action_dim,
            hidden_dim,
            n_heads=state_dim + action_dim,
            init_prob=0.15
        )
        units = hidden_dim * (state_dim + action_dim)

        self.q_network = nn.Sequential(
            nn.Linear(units, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        x = self.ripple_layer(state_action)
        
        return self.q_network(x)