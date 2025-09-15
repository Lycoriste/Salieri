import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution, TanhTransform, Bernoulli

# --- Neural networks --- #
class SAC_Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        # State encoders
        self.agent_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.target_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # state_dim - 6 because everything else goes in here :)
        aux_dim = max(1, state_dim - 6)
        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.shared_layers = nn.Sequential(
            nn.Linear(64 * 3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.move_head = nn.Linear(128, 1)
        self.turn_mu_head = nn.Linear(128, 1)
        self.turn_log_std_head = nn.Linear(128, 1)

        self.apply(self._init_weights)
        
        # Initialize turn_mu bias to 0 to prevent local minima
        nn.init.constant_(self.turn_mu_head.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, state):
        agent_pos = state[:, :3]
        target_pos = state[:, 3:6]
        aux_features = state[:, 6:]
        
        agent_feat = self.agent_encoder(agent_pos)
        target_feat = self.target_encoder(target_pos)
        aux_feat = self.aux_encoder(aux_features)
        
        x = torch.cat([agent_feat, target_feat, aux_feat], dim=1)
        shared = self.shared_layers(x)
        
        move_logits = self.move_head(shared)
        turn_mu = self.turn_mu_head(shared)
        turn_log_std = torch.clamp(self.turn_log_std_head(shared), -10, 2)
        
        return move_logits, turn_mu, turn_log_std

    def get_action_distribution(self, state):
        move_logits, turn_mu, turn_log_std = self.forward(state)
        
        # Discrete movement distribution
        move_dist = Bernoulli(logits=move_logits.squeeze(-1))
        
        # Continuous turning distribution with tanh squashing
        turn_std = torch.exp(turn_log_std)
        normal_dist = Normal(turn_mu.squeeze(-1), turn_std.squeeze(-1))
        turn_dist = TransformedDistribution(normal_dist, TanhTransform())
        
        return move_dist, turn_dist


class SAC_Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        # Same architecture as the actor
        self.agent_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.target_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        aux_dim = max(1, state_dim - 6)
        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Q-value network
        self.q_network = nn.Sequential(
            nn.Linear(64 * 3 + action_dim, 256),
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
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        agent_pos = state[:, :3]
        target_pos = state[:, 3:6]
        aux_features = state[:, 6:]
        
        agent_feat = self.agent_encoder(agent_pos)
        target_feat = self.target_encoder(target_pos)
        aux_feat = self.aux_encoder(aux_features)
        
        state_feat = torch.cat([agent_feat, target_feat, aux_feat], dim=1)
        state_action = torch.cat([state_feat, action], dim=1)
        
        return self.q_network(state_action)
