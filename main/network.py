import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
    
class MultiHead_QN(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        # Process agent and target positions separately
        self.agent_pos_fc = nn.Sequential(
            nn.Linear(3, 32),  # x, y, z
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.target_pos_fc = nn.Sequential(
            nn.Linear(3, 32),  # target x, y, z
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Process angle and distance features (yaw, dist_to_target, dist_to_object)
        self.aux_features_fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Shared deeper feature extractor
        self.shared_fc = nn.Sequential(
            nn.Linear(32 + 32 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Separate heads for action components
        self.move_head = nn.Linear(128, 2)   # Move: forward / no-move
        self.turn_head = nn.Linear(128, 37)   # Turn: left / right / none

    def forward(self, state):
        agent_pos = state[:, 0:3]           # x, y, z
        aux_features = state[:, 3:7]        # yaw_sin, yaw_cos, dist_to_target, dist_to_object
        target_pos = state[:, 7:10]          # target x, y, z 

        # Feature extractors
        agent_feat = self.agent_pos_fc(agent_pos)
        target_feat = self.target_pos_fc(target_pos)
        aux_feat = self.aux_features_fc(aux_features)

        # Concatenate features
        x = torch.cat([agent_feat, target_feat, aux_feat], dim=1)

        # Shared layers
        x = self.shared_fc(x)
        move_q = self.move_head(x)
        turn_q = self.turn_head(x)

        return move_q, turn_q

class MultiHead_A2C(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Process agent and target positions separately
        self.agent_pos_fc = nn.Sequential(
            nn.Linear(3, 32),  # x, y, z
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.target_pos_fc = nn.Sequential(
            nn.Linear(3, 32),  # target x, y, z
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Process angle and distance features (rel pos xyz, yaw_cos, yaw_sin, alignment, dist_to_target)
        self.aux_features_fc = nn.Sequential(
            nn.Linear(state_dim - 6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Shared deeper feature extractor
        self.shared_fc = nn.Sequential(
            nn.Linear(32*3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Discrete actor head for movement
        self.move_logit = nn.Linear(128, 1)

        # Continuous actor head for steering
        self.turn_mu = nn.Linear(128, 1)
        nn.init.zeros_(self.turn_mu.bias)
        nn.init.xavier_uniform_(self.turn_mu.weight, gain=0.1)

        self.turn_log_std = nn.Linear(128, 1)
        nn.init.constant_(self.turn_log_std.bias, -0.5)

        # Critic
        self.value = nn.Linear(128, 1)

    def forward(self, state):
        # Split and process features
        agent_pos = state[:, 0:3]
        target_pos = state[:, 3:6]
        aux_features = state[:, 6:]

        agent_feat = self.agent_pos_fc(agent_pos)
        target_feat = self.target_pos_fc(target_pos)
        aux_feat = self.aux_features_fc(aux_features)

        x = torch.cat([agent_feat, target_feat, aux_feat], dim=1)
        x = self.shared_fc(x)

        # Actor outputs
        move_logit = self.move_logit(x)
        turn_mu = torch.tanh(self.turn_mu(x)) * 0.5
        turn_log_std = torch.clamp(self.turn_log_std(x), min=-2.0, max=1.0)
        turn_std = torch.exp(turn_log_std)

        # Critic output
        value = self.value(x)

        return move_logit, turn_mu, turn_std, value

class Simple_A2C(nn.Module):
    def __init__(self, state_dim: int = 3, action_dim: int = 2):
        super().__init__()
        # A simple shared body to extract features from the state.
        self.shared_body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.move_logit = nn.Linear(64, 1)
        self.turn_mu = nn.Linear(64, 1)
        self.turn_log_std = nn.Linear(64, 1)
        
        self.value = nn.Linear(64, 1)

        # Sensible initializations for the turn action head
        nn.init.zeros_(self.turn_mu.bias)
        nn.init.xavier_uniform_(self.turn_mu.weight, gain=0.1)
        nn.init.constant_(self.turn_log_std.bias, -0.5)

    def forward(self, state):
        features = self.shared_body(state)
        move_logit = self.move_logit(features)
        
        turn_mu = self.turn_mu(features)
        turn_log_std = self.turn_log_std(features)
        turn_std = torch.exp(turn_log_std)
        
        value = self.value(features)

        return move_logit, turn_mu, turn_std, value