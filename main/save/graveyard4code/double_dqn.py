import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math, random
import os, sys, platform
from collections import defaultdict
from network import MultiHead_QN
from replay_memory import ReplayMemory, Transition
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class Double_DQN:
    # Define model with too many hyperparameters
    def __init__(self,
                 state_space: int,
                 batch_size = 128, 
                 gamma = 0.997, 
                 epsilon_start = 0.99, 
                 epsilon_end = 0.1, 
                 epsilon_decay = 1000, 
                 tau = 0.005, 
                 alpha = 1e-4):

        self.batch_size = batch_size
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
        self.memory = ReplayMemory(batch_size * 20)

        # Policy NN + Target NN
        self.policy_net = MultiHead_QN(state_space)
        self.target_net = MultiHead_QN(state_space)

        # Set them to start with the same random weights + biases
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Adam or SGD - whichever stabilizes
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        
        # Current state
        self.state = None
        self.action = None

    def load_policy(self, episode_num: int = 0):
        try:
            self.policy_net.load_state_dict(torch.load(f"save/{episode_num}_Policy_RBX.pth", map_location=device))
            self.target_net.load_state_dict(torch.load(f"save/{episode_num}_Target_RBX.pth", map_location=device))
            self.optimizer.load_state_dict(torch.load(f"save/{episode_num}_Optim_RBX.pth", map_location=device))
            self.policy_net.to(device)
            self.target_net.to(device)

            print("Loaded successfully.")
        except FileNotFoundError as e:
            print(f"No save files found.")

    # Epsilon-greedy action selection with a decaying epsilon as steps -> infinity
    def select_action(self, state):
        # -------------------------------------------------------- #
        # Action space (n = 2)                                     #
        # Moving = 0, 1 (false/true)                               #
        # Steering = [-90,0,90] (turn left, no turn, turn right)     #
        # -------------------------------------------------------- #
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        math.exp(-1.0 * self.steps / self.epsilon_decay)

        if random.random() < eps_threshold:
            # Exploration
            move_action = random.randint(0, 1)
            steer_action = random.randint(0, 36)

            return torch.tensor([move_action, steer_action], device=device, dtype=torch.long)
        else:
            # Exploitation
            with torch.inference_mode(): 
                move_q, steer_q = self.policy_net(state)
                move_action = move_q.argmax(dim=1).item()
                steer_action = steer_q.argmax(dim=1).item()

                return torch.tensor([move_action, steer_action], device=device)
            
    # Optimize agent's neural network
    def optimize_model(self):
        # Can't optimize model when replay memory is not filled
        if (len(self.memory) < self.batch_size):
            return
        # Sample transitions from replay memory
        transitions = self.memory.sample(self.batch_size)
        # Transpose batch: convert batch-array of transitions to transition of batch-arrays
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

        # Get actions from states along the action vector
        action_move = action_batch[:, 0].unsqueeze(1)
        action_steer = action_batch[:, 1].unsqueeze(1)
        q_move, q_steer = self.policy_net(state_batch)

        q_move_selected = q_move.gather(1, action_move)
        q_steer_selected = q_steer.gather(1, action_steer)
        q_total = q_move_selected + q_steer_selected

        # Q(s', a') for all next_states that are not None
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.inference_mode():
            target_move_q, target_steer_q = self.target_net(next_state_batch)
            policy_move_q, policy_steer_q = self.policy_net(next_state_batch)

            # Use policy net to select actions
            next_move_action = policy_move_q.argmax(1, keepdim=True)
            next_steer_action = policy_steer_q.argmax(1, keepdim=True)

            # Use target net to evaluate those actions
            next_move_q_val = target_move_q.gather(1, next_move_action)
            next_steer_q_val = target_steer_q.gather(1, next_steer_action)

            next_state_values[next_state_mask] = (next_move_q_val + next_steer_q_val).squeeze(1)
        
        # If the next_state is None, then the Q(s', a') term evaluates to 0
        expected_action_value = next_state_values * self.gamma + reward_batch

        # Huber loss
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(q_total, expected_action_value.unsqueeze(1))

        # Clear gradients
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping - prevents them from exploding
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 80.0)
        self.optimizer.step()

        # Soft updates
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        
        self.target_net.load_state_dict(target_net_state_dict)

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

        if (done):
            self.episodes_total += 1

        self.optimize_model()
    
    def get_policy(self):
        return self.policy_net, self.target_net
    
    def save_policy(self):
        torch.save(self.policy_net.state_dict(), f"save/double_dqn/{self.episodes_total}_Policy_RBX.pth")
        torch.save(self.target_net.state_dict(), f"save/double_dqn/{self.episodes_total}_Target_RBX.pth")
        torch.save(self.optimizer.state_dict(), f"save/double_dqn/{self.episodes_total}_Optim_RBX.pth")
    
    def get_stats(self):
        print(f"Steps taken: {self.steps}\nEpisodes trained: {self.episodes_total}")
        return self.steps, self.episode_reward, self.episodes_total