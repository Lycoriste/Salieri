import json
import torch
import torch.nn as nn
import torch.F as F

def parse_nn_ac(format: dict, action_heads: dict):
    class Actor(nn.Module):
        def __init__(self, state_dim: int, action_dim: int):
            super().__init__()
            for layer_name, layout in format.items():
                layers = []
                for layer_type, params in layout:
                    layer_cls = getattr(nn, layer_type)
                    layers.append(layer_cls(*params))
                setattr(self, layer_name, nn.Sequential(*layers))

            for action_key, action_dist in action_heads.items():
                setattr(self, action_key, action_dist)

        def forward(self, x):
            for name in format.keys():
                layer = getattr(self, name)
                x = layer(x)
            return x

    class Critic(nn.Module):
        def __init__(self, state_dim: int, action_dim: int):
            super().__init__()
            self.input_dim = state_dim + action_dim

            for layer_name, layout in format.items():
                layers = []
                for layer_type, params in layout:
                    layer_cls = getattr(nn, layer_type)
                    layers.append(layer_cls(*params))
                setattr(self, layer_name, nn.Sequential(*layers))

            for action_key, action_dist in action_heads.items():
                setattr(self, action_key, action_dist)
        
        def forward(self, state, action):
            state_action = torch.cat([state, action], dim=1)
            return self.q_network(state_action)

    return Actor, Critic
