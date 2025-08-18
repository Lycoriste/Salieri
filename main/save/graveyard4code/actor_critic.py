import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, TanhTransform, Bernoulli
import numpy as np
import math, random
import os, sys, platform
from collections import defaultdict
from network import MultiHead_A2C, Simple_A2C
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
                 alpha: float = 1e-4,
                 critic_coef: float = 0.5,
                 entropy_coef: float = 0.01):

        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef

        # Statistics to keep tracking of training progress
        self.steps = 0
        self.episode_reward = defaultdict(int)
        self.episode_length = defaultdict(int)
        self.episodes_total = 0

        # Policy NN
        action_dim = 2
        self.actor_net = Simple_A2C(state_dim, action_dim)

        # Adam or SGD - whichever stabilizes
        self.optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha)
        
        self.ptr = 0
        self.buffer_full = False
        self.state_buf = torch.zeros((n_step, state_dim), dtype=torch.float32, device=device)
        self.action_buf = torch.zeros((n_step, action_dim), dtype=torch.float32, device=device)
        self.reward_buf = torch.zeros((n_step,), dtype=torch.float32, device=device)
        self.done_buf = torch.zeros((n_step,), dtype=torch.float32, device=device)
        
        # State and action handling
        self.state = None
        self.clip_range = 10
        self.action = None

        self.max_turn = math.pi/6

        self.move_p = []
        self.turn_action = []
        self.turn_mu = []
        self.turn_std = []

    def select_action(self, state):
        # Exploitation
        with torch.inference_mode(): 
            move_logit, turn_mu, turn_std, _ = self.actor_net(state)
            move_dist = Bernoulli(logits=move_logit)
            # move_action = move_dist.sample()
            move_action = torch.tensor([0.0], device=device).unsqueeze(1)

            turn_std = torch.clamp(turn_std, min=0.1, max=2.0)
            
            turn_dist = Normal(turn_mu, turn_std)
            turn_dist = TransformedDistribution(turn_dist, TanhTransform())
            turn_action = turn_dist.rsample()

            max_turn = self.max_turn  # 22 degrees in radians
            turn_action_scaled = turn_action * max_turn

            move_logp = move_logp = move_dist.log_prob(move_action)
            turn_logp = turn_dist.log_prob(turn_action).sum(dim=-1)

            action = torch.cat([move_action, turn_action_scaled]).to(device).squeeze()

            self.move_p.append(move_logp.item())
            self.turn_mu.append(turn_mu.item())
            self.turn_std.append(turn_std.item())
            self.turn_action.append(turn_action_scaled.item())

            return action

    # Update da weightss
    def optimize_model(self, next_state_tensor, last_done):
        states = self.state_buf
        actions = self.action_buf
        rewards = self.reward_buf
        dones = self.done_buf

        move_actions = actions[:, 0]
        turn_actions = actions[:, 1]

        move_logit, turn_mu, turn_std, values = self.actor_net(states)
        move_dist = torch.distributions.Bernoulli(logits=move_logit.squeeze())

        turn_std = torch.clamp(turn_std.expand_as(turn_mu).squeeze(-1), min=0.1, max=2.0)
        norm_turn_dist = torch.distributions.Normal(turn_mu.squeeze(-1), turn_std)
        turn_dist = TransformedDistribution(norm_turn_dist, TanhTransform())
        turn_action_unscaled = torch.clamp(turn_actions / self.max_turn, -1.0 + 1e-6, 1.0 - 1e-6)

        # move_log_probs = move_dist.log_prob(move_actions)
        move_log_probs = 0
        turn_log_probs = turn_dist.log_prob(turn_action_unscaled).squeeze()
        logps = move_log_probs + turn_log_probs

        with torch.no_grad():
            next_value = self.actor_net(next_state_tensor)[-1].squeeze() * (1 - last_done)
            returns = torch.zeros_like(rewards, device=device)
            R = next_value

            for i in reversed(range(self.n_step)):
                R = rewards[i] + self.gamma * R * (1 - dones[i])
                returns[i] = R

        values = values.squeeze(-1)
        advantages = returns - values
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(logps * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        
        # move_entropy = move_dist.entropy().mean()
        move_entropy = 0
        turn_entropy = norm_turn_dist.entropy().mean()
        total_entropy = move_entropy + turn_entropy

        turn_bias_penalty = torch.mean(turn_mu ** 2) * 0.01

        loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * total_entropy + turn_bias_penalty

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=0.5)
        self.optimizer.step()

    # Observe state and decide optimal action
    def observe(self, data):
        state = data
        state_tensor = torch.tensor([state], device=device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        action_tensor = self.select_action(state_tensor)
        action = action_tensor.tolist()

        self.state_buf[self.ptr] = state_tensor.squeeze(0)
        self.action_buf[self.ptr] = action_tensor.squeeze(0)

        return action
    
    # From our previous observation, learn from the transition between state to next_state
    def learn(self, data):
        next_state, reward, done = data
        reward_tensor = torch.tensor([reward], device=device)

        self.reward_buf[self.ptr] = reward_tensor
        self.done_buf[self.ptr] = float(done)

        self.episode_reward[self.episodes_total] += reward
        self.episode_length[self.episodes_total] += 1
        self.steps += 1

        if (done):
            self.episodes_total += 1

        self.ptr = (self.ptr + 1) % self.n_step
        if self.ptr == 0:
            self.buffer_full = True

        if self.buffer_full:
            next_state_tensor = torch.tensor([next_state], device=device)
            if next_state_tensor.dim() == 1:
                next_state_tensor = next_state_tensor.unsqueeze(0)
            self.optimize_model(next_state_tensor, done)

    # def normalize_state(self, state):
    #     if not self.is_normalize_state:
    #         return state
            
    #     # Update running statistics during training
    #     if self.actor_net.training:
    #         if isinstance(state, torch.Tensor):
    #             state_np = state.cpu().numpy()
    #         else:
    #             state_np = np.array(state)
                
    #         if state_np.ndim == 1:
    #             state_np = state_np.reshape(1, -1)
                
    #         self.state_normalizer.update(state_np)

    #     mean = torch.tensor(self.state_normalizer.mean, dtype=torch.float32, device=device)
    #     std = torch.tensor(np.sqrt(self.state_normalizer.var + 1e-8), dtype=torch.float32, device=device)
        
    #     if isinstance(state, torch.Tensor):
    #         normalized = (state - mean) / std
    #     else:
    #         state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
    #         normalized = (state_tensor - mean) / std
            
    #     # Clip to prevent extreme values
    #     normalized = torch.clamp(normalized, -self.clip_range, self.clip_range)
    #     return normalized
    
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
            self.actor_net.load_state_dict(torch.load(f"save/a2c/{episode_num}_Policy_RBX.pth", map_location=device))
            self.optimizer.load_state_dict(torch.load(f"save/a2c/{episode_num}_Optim_RBX.pth", map_location=device))
            self.actor_net.to(device)

            print("Loaded successfully.")
        except FileNotFoundError as e:
            print(f"No save files found.")

    def load_metadata_json(self, path: str = "save/a2c/metadata.json"):
        try:
            with open(path, 'r') as f:
                metadata = json.load(f)

            self.steps = metadata.get('steps', 0)
            self.episodes_total = metadata.get('episodes_total', 0)

            if 'episode_length' in metadata:
                episode_length_int_keys = {int(k): v for k, v in metadata['episode_length'].items()}
                self.episode_length = defaultdict(int, episode_length_int_keys)

            if 'episode_reward' in metadata:
                episode_reward_int_keys = {int(k): v for k, v in metadata['episode_reward'].items()}
                self.episode_reward = defaultdict(int, episode_reward_int_keys)
                
            print(f"Metadata loaded. Resuming from {self.episodes_total} episodes and {self.steps} steps.")

        except FileNotFoundError:
            print("No metadata file found. Starting from scratch.")
        except Exception as e:
            print(f"Error loading metadata: {e}. Starting from scratch.")

# Tracks running mean and standard deviation
# class RunningMeanStd:
#     def __init__(self, shape):
#         self.mean = np.zeros(shape, dtype=np.float32)
#         self.var = np.ones(shape, dtype=np.float32)
#         self.count = 1e-4

#     def update(self, x):
#         batch_mean = np.mean(x, axis=0)
#         batch_var = np.var(x, axis=0)
#         batch_count = x.shape[0]
#         self.update_from_moments(batch_mean, batch_var, batch_count)

#     def update_from_moments(self, batch_mean, batch_var, batch_count):
#         delta = batch_mean - self.mean
#         tot_count = self.count + batch_count

#         new_mean = self.mean + delta * batch_count / tot_count
#         m_a = self.var * self.count
#         m_b = batch_var * batch_count
#         M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
#         new_var = M2 / tot_count

#         self.mean = new_mean
#         self.var = new_var
#         self.count = tot_count
