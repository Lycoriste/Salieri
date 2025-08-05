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
        # Shared feature extractor
        self.shared_fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # Separate heads for each action component
        self.move_head = nn.Linear(128, 2)
        self.turn_head = nn.Linear(128, 3)

    def forward(self, x):
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
