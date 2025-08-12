import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, TanhTransform
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
                 alpha: float = 1e-3):

        self.n_step = n_step
        self.gamma = gamma
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
        
        # State and action handling
        self.state = None
        self.state_normalizer = RunningMeanStd(shape=(state_dim,))
        self.is_normalize_state = True
        self.clip_range = 10
        self.action = None

        self.move_p = []
        self.turn_action = []
        self.turn_mu = []
        self.turn_std = []

    def select_action(self, state):
        # Exploitation
        with torch.inference_mode(): 
            move_logit, turn_mu, turn_std, critic_val = self.actor_net(state)
            move_prob = torch.sigmoid(move_logit).squeeze()
            move_action = torch.bernoulli(move_prob).float()
            
            turn_dist = Normal(turn_mu.squeeze(), turn_std.squeeze())
            turn_dist = TransformedDistribution(turn_dist, TanhTransform())
            turn_action = turn_dist.rsample()

            max_turn = math.pi / 8  # 90 degrees in radians
            turn_action_scaled = turn_action * max_turn

            eps = 1e-8
            move_logp = (move_action * torch.log(move_prob + eps) + (1 - move_action) * torch.log(1 - move_prob + eps))
            turn_logp = turn_dist.log_prob(turn_action).sum(dim=-1)
            logp = move_logp + turn_logp

            critic_val = critic_val.squeeze().detach()
            action = torch.stack([move_action, turn_action_scaled]).to(device).squeeze()

            self.move_p.append(move_prob.item())
            self.turn_mu.append(turn_mu.item())
            self.turn_std.append(turn_std.item())
            self.turn_action.append(turn_action_scaled.item())

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
        turn_dist = TransformedDistribution(turn_dist, TanhTransform())
        turn_action_unscaled = turn_actions / (math.pi / 8)
        turn_logp = turn_dist.log_prob(turn_action_unscaled).sum(dim=-1)
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
        state_tensor = self.normalize_state(state)
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

    def normalize_state(self, state):
        if not self.is_normalize_state:
            return state
            
        # Update running statistics during training
        if self.actor_net.training:
            if isinstance(state, torch.Tensor):
                state_np = state.cpu().numpy()
            else:
                state_np = np.array(state)
                
            if state_np.ndim == 1:
                state_np = state_np.reshape(1, -1)
                
            self.state_normalizer.update(state_np)

        mean = torch.tensor(self.state_normalizer.mean, dtype=torch.float32, device=device)
        std = torch.tensor(np.sqrt(self.state_normalizer.var + 1e-8), dtype=torch.float32, device=device)
        
        if isinstance(state, torch.Tensor):
            normalized = (state - mean) / std
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            normalized = (state_tensor - mean) / std
            
        # Clip to prevent extreme values
        normalized = torch.clamp(normalized, -self.clip_range, self.clip_range)
        return normalized
    
    def train(self):
        self.actor_net.train()
    
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
    
    def get_stats(self):
        print(f"Steps taken: {self.steps}\nEpisodes trained: {self.episodes_total}")
        return self.steps, self.episode_reward, self.episodes_total

# Tracks running mean and standard deviation
class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
