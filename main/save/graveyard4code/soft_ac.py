import os, sys, platform
import numpy as np
import math, random
from collections import defaultdict
from replay_memory import PriorityReplay, Transition

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, TanhTransform, Bernoulli

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as io
import json

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class SoftAC:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 batch_size: int = 128,
                 replay_capacity: int = 10000,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 critic_coef: float = 1,
                 entropy_coef: float = 0.05,
        ):
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef

        self.actor = MultiHead_SAC(state_dim, action_dim).to(device)
        self.critic_a = MultiHead_SAC(state_dim, action_dim).to(device)
        self.critic_b = MultiHead_SAC(state_dim, action_dim).to(device)

        # Target networks for stability
        self.critic_a_target = MultiHead_SAC(state_dim, action_dim).to(device)
        self.critic_b_target = MultiHead_SAC(state_dim, action_dim).to(device)

        # Initialize to same weights
        self.critic_a_target.load_state_dict(self.critic_a.state_dict())
        self.critic_b_target.load_state_dict(self.critic_b.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_a.parameters()) + list(self.critic_b.parameters()), lr=critic_lr
        )

        self.batch_size = batch_size
        self.memory = PriorityReplay(replay_capacity)
        
        self.steps = 0
        self.episode_reward = defaultdict(int)
        self.episode_length = defaultdict(int)
        self.episodes_total = 0

        self.state = None
        self.action = None

        self.max_turn = math.pi/4

    def select_action(self, state):
        with torch.inference_mode():
            params = self.actor(state)
            move_dist, turn_dist = MultiHead_SAC.action_dist(params)
            
            move_action = move_dist.sample()
            turn_action = turn_dist.rsample()

            return torch.cat([move_action.unsqueeze(0), turn_action], dim=1).to(device)

    def observe(self, state):
        state_tensor = torch.tensor([state], device=device)
        action_tensor = self.select_action(state_tensor)
        
        self.state = state_tensor
        self.action = action_tensor
        action = action_tensor.squeeze(0).tolist()

        return [action[0], action[1] * self.max_turn]

    def learn(self, data):
        next_state, reward, done = data

        reward = float(reward) 
        done = bool(done)

        next_state_tensor = torch.tensor([next_state], device=device)
        reward_tensor = torch.tensor([reward], device=device)

        package = (self.state, self.action, next_state_tensor, reward_tensor, done)

        if self.state is not None and next_state is not None:
            self.memory.push(*package)

        self.episode_reward[self.episodes_total] += reward
        self.episode_length[self.episodes_total] += 1
        self.steps += 1

        if (done):
            self.episodes_total += 1

        self.optimize()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        experiences, indices, weights  = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*experiences))

        # Filter out all the next_state that are the endpoints of an episode
        next_state_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                        device=device, 
                                        dtype=torch.bool)

        # Concatenates each of the transitions (state, action, reward) into their respective batches
        state_batch = torch.cat(batch.state)
        if state_batch.dim() == 1:
            state_batch = state_batch.unsqueeze(0)

        next_state_batch = torch.cat([s for s in batch.next_state if s is not None])
        if next_state_batch.dim() == 1:
            next_state_batch = next_state_batch.unsqueeze(0)

        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Update critic network
        with torch.no_grad():
            params = self.actor(next_state_batch)
            move_dist, tanh_turn_dist = MultiHead_SAC.action_dist(params)

            next_move_actions = move_dist.sample()
            next_turn_action = tanh_turn_dist.rsample()

            move_log_probs = move_dist.log_prob(next_move_actions).squeeze()
            turn_log_probs = tanh_turn_dist.log_prob(next_turn_action).squeeze()
            total_log_probs = move_log_probs + turn_log_probs

            next_actions = torch.cat([next_move_actions.unsqueeze(1), next_turn_action], dim=1).to(device)

            q_a = self.critic_a_target(next_state_batch, next_actions)
            q_b = self.critic_b_target(next_state_batch, next_actions)

            q_min = torch.min(q_a, q_b).squeeze(-1) 
            q_target = reward_batch + self.gamma * (q_min - self.entropy_coef * total_log_probs)
            
        current_q_a = self.critic_a(state_batch, action_batch).squeeze(-1) 
        current_q_b = self.critic_b(state_batch, action_batch).squeeze(-1) 

        # Critic loss
        loss = F.mse_loss(current_q_a, q_target) + F.mse_loss(current_q_b, q_target)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # Updating actor network
        params = self.actor(state_batch)
        move_dist, tanh_turn_dist = MultiHead_SAC.action_dist(params)
        move_action = move_dist.sample()
        turn_action = tanh_turn_dist.rsample()
        action = torch.cat([move_action.unsqueeze(1), turn_action], dim=1).to(device)

        # Actor loss
        q_a = self.critic_a(state_batch, action)
        q_b = self.critic_b(state_batch, action)
        q_min = torch.min(q_a, q_b)

        log_probs = move_dist.log_prob(move_action).squeeze() + tanh_turn_dist.log_prob(turn_action).squeeze()
        actor_loss = (self.entropy_coef * log_probs - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft updates
        self._soft_update(self.critic_a, self.critic_a_target)
        self._soft_update(self.critic_b, self.critic_b_target)
    
    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    # --- Utility --- #

    def save_policy(self):
        torch.save(self.actor.state_dict(), f"save/sac/{self.episodes_total}_Actor.pth")
        torch.save(self.critic_a.state_dict(), f"save/sac/{self.episodes_total}_CriticA.pth")
        torch.save(self.critic_b.state_dict(), f"save/sac/{self.episodes_total}_CriticB.pth")
        torch.save(self.actor_optimizer.state_dict(), f"save/sac/{self.episodes_total}_ActorOptim.pth")

    def load_policy(self, episode_num: int = 0):
        try:
            self.actor.load_state_dict(torch.load(f"save/sac/{episode_num}_Actor.pth", map_location=device))
            self.critic_a.load_state_dict(torch.load(f"save/sac/{episode_num}_CriticA.pth", map_location=device))
            self.critic_b.load_state_dict(torch.load(f"save/sac/{episode_num}_CriticB.pth", map_location=device))
            self.actor_optimizer.load_state_dict(torch.load(f"save/sac/{episode_num}_ActorOptim.pth", map_location=device))
            self.actor.to(device)
            self.critic_a.to(device)
            self.critic_b.to(device)

            print("Loaded successfully.")
        except FileNotFoundError as e:
            print(f"No save files found:\n", str(e))

# --- Neural network --- #

class MultiHead_SAC(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.agent_nn = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.target_nn = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.aux_nn = nn.Sequential(
            nn.Linear(state_dim - 6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Shared deeper feature extractor
        self.shared_fc = nn.Sequential(
            nn.Linear(32 * 3, 256),
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
        torch.nn.init.constant_(self.turn_mu.bias, 0)

        self.turn_log_std = nn.Linear(128, 1)

        # Critic
        self.q_value = nn.Linear(128 + action_dim, 1)

    def forward(self, state, action=None):
        agent_pos = state[:, 0:3]
        target_pos = state[:, 3:6]
        aux = state[:, 6:]

        agent_feat = self.agent_nn(agent_pos)
        target_feat = self.target_nn(target_pos)
        aux_feat = self.aux_nn(aux)

        x = torch.cat([agent_feat, target_feat, aux_feat], dim=1).to(device)
        shared = self.shared_fc(x)

        move_logit = self.move_logit(shared)
        turn_mu = self.turn_mu(shared)
        turn_log_std = self.turn_log_std(shared)

        if action is None:
            return move_logit, turn_mu, turn_log_std
        else:
            combined = torch.cat([shared, action], dim=1).to(device)
            return self.q_value(combined)
    
    @staticmethod
    def action_dist(payload):
        move_logit, turn_mu, turn_log_std = payload
        move_dist = Bernoulli(logits=move_logit.squeeze(-1))

        turn_std = torch.exp(turn_log_std)
        norm_turn_dist = Normal(turn_mu, turn_std)
        tanh_turn_dist = TransformedDistribution(norm_turn_dist, TanhTransform())

        return move_dist, tanh_turn_dist
