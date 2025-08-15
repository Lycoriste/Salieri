import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, TanhTransform, Bernoulli
import numpy as np
import math, random
import os, sys, platform
from collections import defaultdict
from network import MultiHead_SAC
from replay_memory import PriorityReplay, Transition
import matplotlib.pyplot as plt
import json

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class SoftAC:
    def __init__(self,
                 state_dim: int,
                 batch_size: int,
                 replay_capacity: int,
                 gamma: float,
                 alpha: float,
                 critic_coef: float,
                 entropy_coef: float):
        self.state_dim = state_dim
        self.gamma = gamma
        self.alpha = alpha
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef

        self.actor_network = MultiHead_SAC
        self.batch_size = batch_size
        self.memory = PriorityReplay(replay_capacity)

        self.steps = 0
        self.episode_reward = defaultdict(int)
        self.episode_length = defaultdict(int)
        self.episodes_total = 0

        self.optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha)
        self.state = None
        self.action = None

        self.max_turn = math.pi/4

    def select_action(self, state):
        with torch.inference_mode():
            move_logit, turn_mu, turn_std, _ = self.actor_network(state)
            move_dist = Bernoulli(logits=move_logit)
            move_action = move_dist.sample()

            turn_dist = Normal(turn_mu, turn_std)
            turn_dist = TransformedDistribution(turn_dist, TanhTransform)
            turn_action = turn_dist.rsample()
            turn_action *= self.max_turn

            return torch.cat([move_action, turn_action], device=device)

    def observe(self, data):
        state = data
        state_tensor = torch.tensor([state], device=device)
        action_tensor = self.select_action(state_tensor)
        
        self.state = state_tensor
        self.action = action_tensor.to_list()

        return self.action

    def learn(self, data):
        next_state, reward, done = data

        next_state_tensor = torch.tensor([next_state], device=device)
        reward_tensor = torch.tensor([reward], device=device)

        package = (self.state, self.action, reward_tensor, next_state_tensor, done)

        if self.state is not None and next_state is not None:
            self.memory.push(package)

        self.episode_reward[self.episodes_total] += reward
        self.episode_length[self.episodes_total] += 1
        self.steps += 1

        if (done):
            self.episodes_total += 1

        self.optimize()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

    def save_policy(self):
        torch.save(self.policy_net.state_dict(), f"save/sac/{self.episodes_total}_Policy_RBX.pth")
        torch.save(self.target_net.state_dict(), f"save/sac/{self.episodes_total}_Target_RBX.pth")
        torch.save(self.optimizer.state_dict(), f"save/sac/{self.episodes_total}_Optim_RBX.pth")

    def load_policy(self, episode_num: int = 0):
        try:
            self.policy_net.load_state_dict(torch.load(f"save/sac/{episode_num}_Policy_RBX.pth", map_location=device))
            self.target_net.load_state_dict(torch.load(f"save/sac/{episode_num}_Target_RBX.pth", map_location=device))
            self.optimizer.load_state_dict(torch.load(f"save/sac/{episode_num}_Optim_RBX.pth", map_location=device))
            self.policy_net.to(device)
            self.target_net.to(device)
            print("Loaded successfully.")
        except FileNotFoundError as e:
            print(f"No save files found:\n", str(e))