import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math, random
import os, sys, platform
from collections import defaultdict
from network import MultiHead_A2C
import matplotlib.pyplot as plt
import json

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class ActorCritic:
    def __init__(self,
                 state_dim: int,
                 n_step: int = 5,
                 gamma: float = 0.997, 
                 epsilon_start: float = 0.99, 
                 epsilon_end: float = 0.1, 
                 epsilon_decay: float = 1000.0, 
                 tau: float = 0.005, 
                 alpha: float = 1e-3):

        self.n_step = n_step
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.alpha = alpha

        # Statistics to keep tracking of training progress
        self.steps = 0
        self.episode_reward = defaultdict(int)
        self.episode_length = defaultdict(int)
        self.episodes_total = 0

        # Policy NN
        action_dim = 2
        self.actor_net = MultiHead_A2C(state_dim, action_dim)

        # Adam or SGD - whichever stabilizes
        self.optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha)
        
        self.ptr = 0
        self.state_buf = torch.zeros((n_step, state_dim), dtype=torch.float32, device=device)
        self.action_buf = torch.zeros((n_step, action_dim), dtype=torch.float32, device=device)
        self.reward_buf = torch.zeros((n_step,), dtype=torch.float32, device=device)
        self.done_buf = torch.zeros((n_step,), dtype=torch.float32, device=device)
        self.val_buf = torch.zeros((n_step,), dtype=torch.float32, device=device)
        self.logp_buf = torch.zeros((n_step,), dtype=torch.float32, device=device)
        
        # Current state
        self.state = None
        self.action = None

    def load_policy(self, episode_num: int = 0):
        try:
            self.actor_net.load_state_dict(torch.load(f"save/{episode_num}_Policy_RBX.pth", map_location=device))
            self.optimizer.load_state_dict(torch.load(f"save/{episode_num}_Optim_RBX.pth", map_location=device))
            self.actor_net.to(device)

            print("Loaded successfully.")
        except FileNotFoundError as e:
            print(f"No save files found.")

    def load_metadata_json(self, path: str = "save/a2c/metadata.json"):
        with open(path, 'r') as f:
            metadata = json.load(f)

        self.steps = metadata['steps']
        self.episode_length = defaultdict(int, metadata['episode_length'])
        self.episode_reward = defaultdict(int, metadata['episode_reward'])
        self.episodes_total = metadata['episodes_total']

        return metadata

    def select_action(self, state):
        # Exploitation
        with torch.inference_mode(): 
            move_logit, turn_mu, turn_std, critic_val = self.actor_net(state)
            move_prob = torch.sigmoid(move_logit).squeeze()
            move_action = torch.bernoulli(move_prob).float()
            
            turn_dist = torch.distributions.Normal(turn_mu.squeeze(), turn_std.squeeze())
            turn_action = turn_dist.sample()

            max_turn = math.pi / 2  # 90 degrees in radians
            turn_action = torch.clamp(turn_action, -max_turn, max_turn)

            eps = 1e-8
            move_logp = (move_action * torch.log(move_prob + eps) + (1 - move_action) * torch.log(1 - move_prob + eps))
            turn_logp = turn_dist.log_prob(turn_action).sum(dim=-1)
            logp = move_logp + turn_logp

            critic_val = critic_val.squeeze().detach()
            action = torch.stack([move_action, turn_action]).to(device).squeeze()

            return action, critic_val, logp

    # Optimize agent's neural network
    def optimize_model(self):
        if self.ptr == 0:
            return
            
        states = self.state_buf[:self.ptr]
        actions = self.action_buf[:self.ptr]
        rewards = self.reward_buf[:self.ptr]
        dones = self.done_buf[:self.ptr]

        move_logit, turn_mu, turn_std, values = self.actor_net(states)
        move_prob = torch.sigmoid(move_logit).squeeze(-1)
        move_actions = actions[:, 0]
        turn_actions = actions[:, 1]
        
        eps = 1e-8
        move_logp = move_actions * torch.log(move_prob + eps) + (1 - move_actions) * torch.log(1 - move_prob + eps)
        turn_std = turn_std.expand_as(turn_mu).squeeze(-1)
        turn_dist = torch.distributions.Normal(turn_mu.squeeze(-1), turn_std)
        turn_logp = turn_dist.log_prob(turn_actions).sum(dim=-1)
        logps = move_logp + turn_logp

        returns = torch.zeros(self.ptr, device=device)
        R = 0
        for i in reversed(range(self.ptr)):
            if dones[i]:
                R = 0
            R = rewards[i] + self.gamma * R
            returns[i] = R

        advantages = (returns - values.squeeze(-1)).detach()
        critic_loss = F.mse_loss(values.squeeze(-1), returns)

        actor_loss = -(logps * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=0.5)
        self.optimizer.step()


    # Observe state and decide optimal action
    def observe(self, data):
        if (self.ptr >= self.n_step):
            self.ptr = 0

        state = data
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        action_tensor, critic_value, logp = self.select_action(state_tensor)
        action = action_tensor.tolist()

        self.state_buf[self.ptr] = state_tensor.squeeze(0)
        self.action_buf[self.ptr] = action_tensor.squeeze(0)
        self.val_buf[self.ptr] = critic_value.squeeze(0)
        self.logp_buf[self.ptr] = logp.squeeze(0)

        return action
    
    # From our previous observation, learn from the transition between state to next_state
    def learn(self, data):
        if (self.ptr >= self.n_step):
            self.optimize_model()
            self.ptr = 0

        next_state, reward, done = data
        reward_tensor = torch.tensor([reward], device=device)

        self.reward_buf[self.ptr] = reward_tensor
        self.done_buf[self.ptr] = done

        self.episode_reward[self.episodes_total] += reward
        self.episode_length[self.episodes_total] += 1
        self.steps += 1

        if (done):
            self.episodes_total += 1
        
        self.ptr += 1
    
    def get_policy(self):
        return self.actor_net
    
    def save_policy(self):
        torch.save(self.actor_net.state_dict(), f"save/a2c/{self.episodes_total}_Policy_RBX.pth")
        torch.save(self.optimizer.state_dict(), f"save/a2c/{self.episodes_total}_Optim_RBX.pth")

        metadata = {
            "steps": self.steps,
            "episode_length": dict(self.episode_length),
            "episode_reward": dict(self.episode_reward),
            "episodes_total": self.episodes_total
        }
        with open("save/a2c/metadata.json", 'w') as f:
            json.dump(metadata, f)
    
    def get_stats(self):
        print(f"Steps taken: {self.steps}\nEpisodes trained: {self.episodes_total}")
        return self.steps, self.episode_reward, self.episodes_total