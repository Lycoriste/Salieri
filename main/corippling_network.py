from experimental import AdaptiveCorippleLayer

import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution, TanhTransform, Bernoulli

# --- Experimental corippling networks --- #
class COR_Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.units = hidden_dim * action_dim

        self.ripple_layer_1 = AdaptiveCorippleLayer(
            state_dim,
            hidden_dim,
            n_heads=action_dim,
            init_prob=0.5
        )

        self.ripple_layer_2 = AdaptiveCorippleLayer(
            self.units,
            hidden_dim,
            n_heads=action_dim,
            init_prob=0.5
        )


        self.move_head = nn.Linear(self.units, 1)
        self.turn_mu_head = nn.Linear(self.units, 1)
        self.turn_log_std_head = nn.Linear(self.units, 1)

        self.apply(self._init_weights)
        
        # Initialize turn_mu bias to 0 to prevent local minima
        nn.init.constant_(self.turn_mu_head.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, state):
        # print("STATE:", state)
        x = self.ripple_layer_1(state)
        x = self.ripple_layer_2(x)
        
        move_logits = self.move_head(x)
        # print("MOVE_LOGITS (pre-clamp):", move_logits)
        turn_mu = self.turn_mu_head(x)
        # print("TURN_MU (pre-clamp):", turn_mu)
        turn_log_std = self.turn_log_std_head(x)
        # print("TURN_LOG_STD (pre-clamp):", turn_log_std)

        # Clamp for stability
        move_logits = torch.clamp(move_logits, -10, 10)
        turn_log_std = torch.clamp(turn_log_std, -10, 2)
        
        return move_logits, turn_mu, turn_log_std

    # Highly aggressive clamping for exploding gradients from ACL
    def get_action_distribution(self, state):
        state_tensor = torch.tensor(state)
        move_logits, turn_mu, turn_log_std = self.forward(state_tensor)
        
        # Discrete movement distribution
        eps = 1e-6
        move_probs = torch.sigmoid(move_logits).clamp(eps, 1 - eps)
        move_dist = Bernoulli(logits=move_probs.squeeze(-1))
        
        # Clamp turn_mu for stability
        turn_mu = turn_mu.clamp(-3.0, 3.0)

        # Continuous turning distribution with tanh squashing
        min_std = 1e-3
        turn_std = torch.exp(turn_log_std).clamp(min=min_std)
        normal_dist = Normal(turn_mu.squeeze(-1), turn_std.squeeze(-1))
        turn_dist = TransformedDistribution(normal_dist, TanhTransform())
        
        return move_dist, turn_dist


class COR_Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.units = hidden_dim * (state_dim + action_dim)

        self.ripple_layer_1 = AdaptiveCorippleLayer(
            state_dim + action_dim,
            hidden_dim,
            n_heads=state_dim + action_dim,
            init_prob=0.5
        )

        self.ripple_layer_2 = AdaptiveCorippleLayer(
            self.units,
            hidden_dim,
            n_heads=state_dim + action_dim,
            init_prob=0.5
        )

        self.q_network = nn.Sequential(
            nn.Linear(self.units, 256),
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
        x = self.ripple_layer_1(state_action)
        x = self.ripple_layer_2(x)
        
        return self.q_network(x)