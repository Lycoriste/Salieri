import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, TanhTransform, Bernoulli
import numpy as np
import math, random
import os, sys, platform
from collections import defaultdict
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
                 action_dim: int,
                 batch_size: int,
                 replay_capacity: int,
                 gamma: float,
                 alpha_lr: float = 3e-4,
                 tau: float = 1.0,
                 critic_coef: float = 1,
                 entropy_coef: float = 1,
        ):
        self.state_dim = state_dim
        self.gamma = gamma
        self.alpha_lr = alpha_lr
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef

        self.actor = MultiHead_SAC(state_dim, action_dim)
        self.critic_a = MultiHead_SAC(state_dim, action_dim)
        self.critic_b = MultiHead_SAC(state_dim, action_dim)

        # Initialize to same weights
        self.critic_a.load_state_dict(self.actor.state_dict())
        self.critic_b.load_state_dict(self.actor.state_dict())

        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.alpha)
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
            move_logit, turn_mu, turn_std = self.actor(state)
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
        done = bool(done)

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

        # Filter out all the next_state that are the endpoints of an episode
        next_state_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                        device=device, 
                                        dtype=torch.bool)

        # Concatenates each of the transitions (state, action, reward) into their respective batches
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None])

        next_action_batch = self.actor(next_state_batch)
        q_a = self.critic_a(next_state_batch, next_action_batch)
        q_b = self.critic_b(next_state_batch, next_action_batch)

        move_logits = next_action_batch[:, 0]
        turn_mus = next_action_batch[:, 1]
        turn_stds = next_action_batch[:, 2]

        move_dist = Bernoulli(logits=move_logits.squeeze())
        norm_turn_dist = Normal(turn_mus.unsqueeze(1), turn_stds.unsqueeze(1))
        tanh_turn_dist = TransformedDistribution(norm_turn_dist, TanhTransform())

        next_move_actions = move_dist.sample()
        next_turn_actions_unscaled = tanh_turn_dist.rsample()
        next_turn_actions_scaled = next_turn_actions_unscaled * self.max_turn

        move_log_probs = move_dist.log_prob(next_move_actions).squeeze()
        turn_log_probs = tanh_turn_dist.log_prob(next_turn_actions_scaled).squeeze()
        total_log_probs = move_log_probs + turn_log_probs
        
        q_min = torch.min(q_a, q_b)
        q_target = reward_batch + self.gamma * (q_min - self.alpha_lr * total_log_probs)

        current_q_a = self.critic_a(state_batch, action_batch)
        current_q_b = self.critic_b(state_batch, action_batch)

        current_move_log_probs = move_dist.log_prob(action_batch[:, 0]).squeeze()
        current_turn_log_probs = tanh_turn_dist.log_prob(action_batch[:, 1]).squeeze()
        current_total_log_probs = current_move_log_probs + current_turn_log_probs

        current_q_min = torch.min(current_q_a, current_q_b)
        current_q_target = reward_batch + self.gamma * (current_q_min - self.alpha_lr * current_total_log_probs)

        loss = F.mse_loss(current_q_target, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # --- Utility --- #

    def save_policy(self):
        torch.save(self.actor.state_dict(), f"save/sac/{self.episodes_total}_Actor.pth")
        torch.save(self.critic_a.state_dict(), f"save/sac/{self.episodes_total}_CriticA.pth")
        torch.save(self.critic_b.state_dict(), f"save/sac/{self.episodes_total}_CriticB.pth")
        torch.save(self.optimizer.state_dict(), f"save/sac/{self.episodes_total}_Optim.pth")

    def load_policy(self, episode_num: int = 0):
        try:
            self.actor.load_state_dict(torch.load(f"save/sac/{episode_num}_Actor.pth", map_location=device))
            self.critic_a.load_state_dict(torch.load(f"save/sac/{episode_num}_CriticA.pth", map_location=device))
            self.critic_b.load_state_dict(torch.load(f"save/sac/{episode_num}_CriticB.pth", map_location=device))
            self.optimizer.load_state_dict(torch.load(f"save/sac/{episode_num}_Optim.pth", map_location=device))
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
            nn.Linear(32),
            nn.ReLU()
        )

        self.target_nn = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32),
            nn.ReLU()
        )

        # Shared deeper feature extractor
        self.shared_fc = nn.Sequential(
            nn.Linear(32 * state_dim, 256),
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
        self.turn_log_std = nn.Linear(128, 1)

        # Critic
        self.value = nn.Linear(128, 1)

    def forward(self, state, action=None):
        agent_pos = state[:, 0:3]
        target_pos = state[:, 3:6]

        agent_feat = self.agent_nn(agent_pos)
        target_feat = self.target_nn(target_pos)

        x = torch.cat([agent_feat, target_feat], device=device)
        shared = self.shared_fc(x)

        move_logit = self.move_logit(shared)
        turn_mu = self.turn_mu(shared)
        turn_log_std = self.turn_log_std(shared)

        if action is None:
            return move_logit, turn_mu, turn_log_std
        else:
            combined = torch.cat([shared, action], dim=1, device=device)
            return self.value(combined)