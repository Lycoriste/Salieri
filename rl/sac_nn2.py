import torch
import torch.nn as nn
import torch.distributions as dist

class SAC_Actor(nn.Module):
    def __init__(self, state_dim: int, action_params: dict):
        self.layer_1 = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.Linear(128, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
        )
        
        self.actions_head = {}
        for action, params in action_params:
            if params['type'] == 'Normal':
                self.actions_head[action] = (nn.Linear(128, 1), nn.Linear(128, 1))
            else:
                self.actions_head[action] = (nn.Linear(128, 1),)

    def forward(self, state):
        ...

    def get_action_distribution(self, state):
        ...

    def _init_weights(self, m):
        ...
