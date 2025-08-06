import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Basic linear layers for spatial navigation 
class Spatial_DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Spatial_DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class Multi_Head_QN(nn.Module):
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

# DQN for an Atari Breakout Agent
# class Sample_DQN(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(DQN, self).__init__()
#         channel, _, _ = state_dim

#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

#         self.fc1 = nn.Linear(3136, 512)
#         self.fc2 = nn.Linear(512, action_dim)

#     def forward(self, x):
#         # Normalize x -> (0, 255)
#         x /= 255.0
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = torch.flatten(x, start_dim=1)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)
