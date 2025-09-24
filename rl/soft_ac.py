import os, sys, platform
import math, random
from collections import defaultdict
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .replay_memory import PriorityReplay, Transition
from .sac_nn import SAC_Actor, SAC_Critic
import numpy as np

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class SoftAC:
    def __init__(self,
                 state_dim: int,
                 action_design: dict,
                 batch_size: int = 256,
                 replay_capacity: int = 10000,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-4,
                 entropy_coef: float = 0.05,
                 target_update_freq: int = 2,
        ):
        self.state_dim = state_dim
        self.action_dim = self.count_action(action_design)

        self.gamma = gamma
        self.tau = tau
        self.entropy_coef = entropy_coef
        self.target_update_freq = target_update_freq

        actor_network = SAC_Actor
        critic_network = SAC_Critic
        
        self.actor = actor_network(state_dim, action_dim).to(device)
        self.critic_a = critic_network(state_dim, action_dim).to(device)
        self.critic_b = critic_network(state_dim, action_dim).to(device)

        # Target networks for stability
        self.critic_a_target = critic_network(state_dim, action_dim).to(device)
        self.critic_b_target = critic_network(state_dim, action_dim).to(device)

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


    def select_action(self, state, deterministic=False):
        self.actor.eval()

        with torch.no_grad():
            actions = []
            dists = self.actor.get_action_distribution(state)
            
            for dist in dists:
                if deterministic:
                    if hasattr(dist, "mean"):
                        action = dist.mean
                    elif hasattr(dist, "probs"):
                        action = torch.argmax(dist.probs, dim=-1, keepdim=True)
                    else:
                        print("[!] Deterministic action could not be resolved.")
                        action = dist.rsample if dist.has_rsample else dist.sample()
                else:
                    action = dist.rsample() if dist.has_rsample else dist.sample()
                
                if (action.dim() == 1):
                    action = action.unsqueeze(-1)
                actions.append(action)
            
            action = torch.cat(actions, dim=-1)

        self.actor.train()
        return action


    def step(self, state, deterministic=False):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        batch = state_tensor.ndim == 2
        if not batch:
            state_tensor = state_tensor.unsqueeze(0)
        action_tensor = self.select_action(state_tensor, deterministic)
        
        self.state = state_tensor
        self.action = action_tensor
        
        # NOTE Actions should always unscaled on server and scaling should be performed by game
        if not batch:
            action = action_tensor.squeeze(0).tolist()
            return [float(a) for a in action]

        actions = []
        for action in action_tensor:
            action = action.tolist()
            actions.append([float(a) for a in action])

        return actions


    def update(self, data):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = data
        
        # TODO: Needs more testing
        for state, action, next_state, reward, done in zip(state_batch, 
                                                           action_batch, 
                                                           next_state_batch, 
                                                           reward_batch, 
                                                           done_batch):
            state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
            action_tensor = torch.tensor([action], dtype=torch.float32, device=device)
            next_state_tensor = torch.tensor([next_state], dtype=torch.float32, device=device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
            done_tensor = torch.tensor([done], dtype=torch.bool, device=device)

            package = Transition(state_tensor, action_tensor, next_state_tensor, reward_tensor, done_tensor)
            self.memory.push(*package)

        # Update networks and everything breaks
        if len(self.memory) >= self.batch_size:
            self.optimize()


    def optimize(self):
        # TODO: Priority replay unimplemented (!!!)
        self.actor.train()
        self.critic_a.train()
        self.critic_b.train()

        experiences, _, _ = self.memory.sample(self.batch_size)
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

        if torch.isnan(critic_loss):
            print("\033[31m[!] Warning: critic_loss was NaN, skipping update")
            return  # Skip update if loss is NaN

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
            if torch.isnan(actor_loss):
                print("\033[31m[!] Warning: actor_loss was NaN, skipping update")
                return # Skip update if loss is NaN
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update(self.critic_a, self.critic_a_target)
            self._soft_update(self.critic_b, self.critic_b_target)

        self.update_counter += 1


    def _compute_critic_loss(self,
                             state_batch,
                             action_batch,
                             reward_batch,
                             next_state_batch,
                             done_batch
        ):
        with torch.no_grad():
            # Sample next actions from current policy
            next_log_probs, next_actions = self._compute_log_probs(next_state_batch)

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
        log_probs, actions = self._compute_log_probs(state_batch)

        # Compute Q-values
        q1 = self.critic_a(state_batch, actions).squeeze(-1)
        q2 = self.critic_b(state_batch, actions).squeeze(-1)
        q_min = torch.min(q1, q2)

        # Actor loss: maximize Q - alpha * log_prob
        actor_loss = (self.entropy_coef * log_probs - q_min).mean()
        return actor_loss
    
    
    # NOTE Misleading name since it returns log_probs and action_batch
    def _compute_log_probs(self, state_batch):
        dists = self.actor.get_action_distribution(state_batch)
        log_probs = torch.zeros(state_batch.size(0), device=state_batch.device)
        actions = []

        for dist in dists:
            # Use rsample for continuous dists and transformed dists
            action = dist.rsample() if dist.has_rsample else dist.sample()
            if action.dim() == 1:
                action = action.unsqueeze(-1)

            lp = dist.log_prob(action)
            if lp.dim() > 1:
                lp = lp.sum(dim=-1)
            log_probs += lp
            actions.append(action)

        # NOTE Use only if NaN updates are too frequent
        log_probs = torch.clamp(log_probs, min=-10.0, max=10.0)
        action_batch = torch.cat(actions, dim=-1)
        
        return log_probs, action_batch


    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    
    def count_action(self, action_design: dict) -> int:
        count = 0
        PARAM_COUNT = {
           "normal": 2,     # mu + std 
           "continuous": 2, # mu + std 
           "binary": 1,     # logit 
           "logit": 1,      # logit 
        }

        for params in action_design.values():
            t = params["type"].lower()
            if t in PARAM_COUNT:
                count += PARAM_COUNT[t]
            elif t == "categorical":            # unique case
                count += params["n_classes"]
            else:
                raise ValueError(f"Unsupported action type: {t}")
    
        return count


