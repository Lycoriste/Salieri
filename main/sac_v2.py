import os, sys, platform
import numpy as np
import math, random
from collections import defaultdict
from typing import Tuple, Optional
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
                 batch_size: int = 256,
                 replay_capacity: int = 100000,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-4,
                 entropy_coef: float = 0.2,
                 target_update_freq: int = 2,
        ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.entropy_coef = entropy_coef
        self.target_update_freq = target_update_freq

        self.actor = SAC_Actor(state_dim, action_dim).to(device)
        self.critic_a = SAC_Critic(state_dim, action_dim).to(device)
        self.critic_b = SAC_Critic(state_dim, action_dim).to(device)

        # Target networks for stability
        self.critic_a_target = SAC_Critic(state_dim, action_dim).to(device)
        self.critic_b_target = SAC_Critic(state_dim, action_dim).to(device)

        # Initialize target networks with same weights
        self.critic_a_target.load_state_dict(self.critic_a.state_dict())
        self.critic_b_target.load_state_dict(self.critic_b.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic_a.parameters()) + list(self.critic_b.parameters()), 
            lr=critic_lr
        )

        self.batch_size = batch_size
        self.memory = PriorityReplay(replay_capacity)
        
        self.steps = 0
        self.episode_reward = defaultdict(float)
        self.episode_length = defaultdict(int)
        self.episodes_total = 0
        self.update_counter = 0

        self.state = None
        self.action = None

        self.max_turn = math.pi/4

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            if deterministic:
                # For evaluation, use deterministic policy
                move_logits, turn_mu, _ = self.actor(state)
                move_action = (torch.sigmoid(move_logits) > 0.5).float()
                turn_action = torch.tanh(turn_mu)
            else:
                # For training, sample from policy
                move_dist, turn_dist = self.actor.get_action_distribution(state)
                move_action = move_dist.sample()
                turn_action = turn_dist.rsample()
            
            # Dimension checking
            if move_action.dim() == 1:
                move_action = move_action.unsqueeze(-1)
            if turn_action.dim() == 1:
                turn_action = turn_action.unsqueeze(-1)
            
            action = torch.cat([move_action, turn_action], dim=-1)
            return action

    def observe(self, state, deterministic=False):
        state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
        action_tensor = self.select_action(state_tensor, deterministic)
        
        self.state = state_tensor
        self.action = action_tensor
        
       # Return action but scaled
       # NOTE: Turning is always unscaled when training
        action = action_tensor.squeeze(0).tolist()

        return [float(action[0]), float(action[1] * self.max_turn)]

    def learn(self, data):
        next_state, reward, done = data
        reward = float(reward) 
        done = bool(done)

        if self.state is not None:
            next_state_tensor = torch.tensor([next_state], dtype=torch.float32, device=device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
            done_tensor = torch.tensor([done], dtype=torch.bool, device=device)

            # Friendly Amazon package
            package = Transition(self.state, self.action, next_state_tensor, reward_tensor, done_tensor)
            self.memory.push(*package)

        # Update statistics
        self.episode_reward[self.episodes_total] += reward
        self.episode_length[self.episodes_total] += 1
        self.steps += 1

        if done:
            self.episodes_total += 1

        # Update networks and everything breaks
        if len(self.memory) >= self.batch_size:
            self.optimize()

        # Update state for next step
        # NOTE: Need experimenting with this to see if ROBLOX script is bugged
        if not done:
            self.state = torch.tensor([next_state], dtype=torch.float32, device=device)
        else:
            self.state = None

    def optimize(self):
        # TODO: Implement priority
        experiences, indices, weights = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*experiences))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        # Critic loss
        critic_loss = self._compute_critic_loss(
            state_batch, action_batch, reward_batch, 
            next_state_batch, done_batch
        )

        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_a.parameters()) + list(self.critic_b.parameters()), 
            max_norm=1.0
        )
        self.critic_optimizer.step()

        # Update actor periodically for stability
        if self.update_counter % self.target_update_freq == 0:
            actor_loss = self._compute_actor_loss(state_batch)
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update(self.critic_a, self.critic_a_target)
            self._soft_update(self.critic_b, self.critic_b_target)

        self.update_counter += 1

    def _compute_critic_loss(self, state_batch, action_batch, reward_batch, 
                           next_state_batch, done_batch):
        with torch.no_grad():
            # Sample next actions from current policy
            next_move_dist, next_turn_dist = self.actor.get_action_distribution(next_state_batch)
            next_move_actions = next_move_dist.sample()
            next_turn_actions = next_turn_dist.rsample()
            
            # Ensure proper dimensions
            if next_move_actions.dim() == 1:
                next_move_actions = next_move_actions.unsqueeze(-1)
            if next_turn_actions.dim() == 1:
                next_turn_actions = next_turn_actions.unsqueeze(-1)
            
            next_actions = torch.cat([next_move_actions, next_turn_actions], dim=-1)

            # Compute log probabilities for entropy regularization
            next_move_log_probs = next_move_dist.log_prob(next_move_actions.squeeze(-1))
            next_turn_log_probs = next_turn_dist.log_prob(next_turn_actions.squeeze(-1))
            next_log_probs = next_move_log_probs + next_turn_log_probs

            # Clamp log probs because gradients exploded
            next_log_probs = torch.clamp(next_log_probs, -10, 10)

            # Compute target Q-values
            next_q1 = self.critic_a_target(next_state_batch, next_actions)
            next_q2 = self.critic_b_target(next_state_batch, next_actions)
            next_q = torch.min(next_q1, next_q2).squeeze(-1)

            # Target with entropy regularization
            target_q = reward_batch + (1 - done_batch.float()) * self.gamma * (
                next_q - self.entropy_coef * next_log_probs
            )

        # Current Q-values
        current_q1 = self.critic_a(state_batch, action_batch).squeeze(-1)
        current_q2 = self.critic_b(state_batch, action_batch).squeeze(-1)

        # Compute losses
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        return critic_loss

    def _compute_actor_loss(self, state_batch):
        # Sample actions from current policy
        move_dist, turn_dist = self.actor.get_action_distribution(state_batch)
        move_actions = move_dist.sample()  # Use rsample for gradients
        turn_actions = turn_dist.rsample()

        move_actions = move_actions.view(-1, 1)
        turn_actions = turn_actions.view(-1, 1)
        
        actions = torch.cat([move_actions, turn_actions], dim=-1)

        # Compute log probabilities
        move_log_probs = move_dist.log_prob(move_actions)
        turn_log_probs = turn_dist.log_prob(turn_actions)
        log_probs = move_log_probs + turn_log_probs.sum(dim=-1)
        log_probs = torch.clamp(log_probs, -10, 10)

        # Compute Q-values
        q1 = self.critic_a(state_batch, actions).squeeze(-1)
        q2 = self.critic_b(state_batch, actions).squeeze(-1)
        q_min = torch.min(q1, q2)

        # Actor loss: maximize Q - alpha * log_prob
        actor_loss = (self.entropy_coef * log_probs - q_min).mean()
        return actor_loss

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    # Utility functions
    def save_policy(self, path="save/sac/"):
        os.makedirs(path, exist_ok=True)
        
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_a': self.critic_a.state_dict(),
            'critic_b': self.critic_b.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episodes_total': self.episodes_total,
            'steps': self.steps,
        }, f"{path}/checkpoint_{self.episodes_total}.pth")

    def load_policy(self, path="save/sac/", episode_num=None):
        try:
            if episode_num is not None:
                checkpoint_path = f"{path}/checkpoint_{episode_num}.pth"
            else:
                import glob
                checkpoints = glob.glob(f"{path}/checkpoint_*.pth")
                if not checkpoints:
                    raise FileNotFoundError(f"No checkpoints found in {path}")
                checkpoint_path = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic_a.load_state_dict(checkpoint['critic_a'])
            self.critic_b.load_state_dict(checkpoint['critic_b'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

            self.critic_a_target.load_state_dict(self.critic_a.state_dict())
            self.critic_b_target.load_state_dict(self.critic_b.state_dict())
            
            self.episodes_total = checkpoint['episodes_total']
            self.steps = checkpoint['steps']
            
            print(f"Loaded checkpoint from episode {self.episodes_total}")
            
        except FileNotFoundError as e:
            print(f"No checkpoint found: {str(e)}")

# --- Neural networks --- #
class SAC_Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        # State encoders
        self.agent_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.target_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # state_dim - 6 because everything else goes in here :)
        aux_dim = max(1, state_dim - 6)
        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.shared_layers = nn.Sequential(
            nn.Linear(64 * 3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.move_head = nn.Linear(128, 1)
        self.turn_mu_head = nn.Linear(128, 1)
        self.turn_log_std_head = nn.Linear(128, 1)

        self.apply(self._init_weights)
        
        # Initialize turn_mu bias to 0 to prevent local minima
        nn.init.constant_(self.turn_mu_head.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, state):
        agent_pos = state[:, :3]
        target_pos = state[:, 3:6]
        aux_features = state[:, 6:]
        
        agent_feat = self.agent_encoder(agent_pos)
        target_feat = self.target_encoder(target_pos)
        aux_feat = self.aux_encoder(aux_features)
        
        x = torch.cat([agent_feat, target_feat, aux_feat], dim=1)
        shared = self.shared_layers(x)
        
        move_logits = self.move_head(shared)
        turn_mu = self.turn_mu_head(shared)
        turn_log_std = torch.clamp(self.turn_log_std_head(shared), -10, 2)
        
        return move_logits, turn_mu, turn_log_std

    def get_action_distribution(self, state):
        move_logits, turn_mu, turn_log_std = self.forward(state)
        
        # Discrete movement distribution
        move_dist = Bernoulli(logits=move_logits.squeeze(-1))
        
        # Continuous turning distribution with tanh squashing
        turn_std = torch.exp(turn_log_std)
        normal_dist = Normal(turn_mu.squeeze(-1), turn_std.squeeze(-1))
        turn_dist = TransformedDistribution(normal_dist, TanhTransform())
        
        return move_dist, turn_dist


class SAC_Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        # Same architecture as the actor
        self.agent_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.target_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        aux_dim = max(1, state_dim - 6)
        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Q-value network
        self.q_network = nn.Sequential(
            nn.Linear(64 * 3 + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        agent_pos = state[:, :3]
        target_pos = state[:, 3:6]
        aux_features = state[:, 6:]
        
        agent_feat = self.agent_encoder(agent_pos)
        target_feat = self.target_encoder(target_pos)
        aux_feat = self.aux_encoder(aux_features)
        
        state_feat = torch.cat([agent_feat, target_feat, aux_feat], dim=1)
        state_action = torch.cat([state_feat, action], dim=1)
        
        return self.q_network(state_action)