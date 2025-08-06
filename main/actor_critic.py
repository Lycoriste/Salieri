import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math, random
import os, sys, platform
from collections import defaultdict
from network import Multi_Head_QN
from replay_memory import PrioritizedReplayMemory, Transition
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class ActorCritic:
    def __init__(self,
                 state_space: tuple,
                 action_space: tuple,
                 n_step: int = 5,
                 n_envs: int = 1,
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
        self.actor_net = Multi_Head_QN(state_space)

        # Adam or SGD - whichever stabilizes
        self.optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha)
        
        self.ptr = 0
        self.state_buf = torch.zeros((n_step, n_envs, *state_space))
        self.action_buf = torch.zeros((n_step, n_envs, *action_space))
        self.reward_buf = torch.zeros((n_step, n_envs))
        self.done_buf = torch.zeros((n_step, n_envs))
        # For values (critic output)
        self.val_buf = torch.zeros((n_step, n_envs))
        # For log probs (actor output)
        self.logp_buf = torch.zeros((n_step, n_envs))
        
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

    def select_action(self, state):
        # Exploitation
        with torch.inference_mode(): 
            move_logit, turn_mu, turn_std, _ = self.policy_net(state)
            move_prob = torch.sigmoid(move_logit)
            move_action = torch.bernoulli(move_prob)
            
            turn_dist = torch.distributions.Normal(turn_mu, turn_std)
            turn_action = turn_dist.sample()
            return torch.tensor([move_action, turn_action], device=device)

    # Optimize agent's neural network
    def optimize_model(self):
        ...
    
    # Observe state and decide optimal action
    def observe(self, data):
        state = data
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        state_tensor = state_tensor.unsqueeze(0)
        action_tensor = self.select_action(state_tensor)
        action = action_tensor.tolist()
        action_tensor = action_tensor.unsqueeze(0)

        self.state = state_tensor
        self.action = action_tensor

        return action
    
    # From our previous observation, learn from the transition between state to next_state
    def learn(self, data):
        next_state, reward, done = data

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device) 
        next_state_tensor = next_state_tensor.unsqueeze(0)
        reward_tensor = torch.tensor([reward], device=device)

        state_tensor, action_tensor = self.state, self.action
        if state_tensor is not None and next_state is not None:
            self.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor, done)

        self.episode_reward[self.episodes_total] += reward
        self.episode_length[self.episodes_total] += 1
        self.steps += 1
        self.ptr += 1

        if (done):
            self.episodes_total += 1

        if (self.n_step % self.ptr == 0):
            self.optimize_model()
            self.ptr = 0

    def store_transition(self, state, action, reward, done, logp, value):
        self.state_buf[self.ptr].copy_(state)
        self.action_buf[self.ptr].copy_(action)
        self.reward_buf[self.ptr].copy_(reward)
        self.done_buf[self.ptr].copy_(done)
        self.logp_buf[self.ptr].copy_(logp)
        self.val_buf[self.ptr].copy_(value)
        self.ptr += 1
    
    def get_policy(self):
        return self.actor_net
    
    def save_policy(self):
        torch.save(self.actor_net.state_dict(), f"save/double_dqn/{self.episodes_total}_Policy_RBX.pth")
        torch.save(self.optimizer.state_dict(), f"save/double_dqn/{self.episodes_total}_Optim_RBX.pth")
    
    def get_stats(self):
        print(f"Steps taken: {self.steps}\nEpisodes trained: {self.episodes_total}")
        return self.steps, self.episode_reward, self.episodes_total