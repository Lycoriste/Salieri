import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math, random
import os, sys, platform
from collections import defaultdict
from models import Spatial_DQN
from replay_memory import ReplayMemory, Transition
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class RBX_Agent:
    # Define model with too many hyperparameters
    def __init__(self,
                 state_space: int,
                 action_space: int,
                 batch_size = 128, 
                 gamma = 0.99, 
                 epsilon_start = 0.99, 
                 epsilon_end = 0.01, 
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

        # NOTE: Not necessarily batch_size * 1000 - it's just what worked for me
        # We use experience replay to stabilize training through random sampling
        self.memory = ReplayMemory(batch_size * 1000)

        # Policy NN + Target NN
        self.policy_net = Spatial_DQN(state_space, action_space)
        self.target_net = Spatial_DQN(state_space, action_space)

        # Set them to start with the same random weights + biases
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Adam or SGD - whichever stabilizes
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        
        # Current state
        self.state = None
        self.action = None

    # Epsilon-greedy action selection with a decaying epsilon as steps -> infinity
    def select_action(self, state):
        # Epsilon Threshold: Early in training -> prioritize exploring | Later in training -> exploit more
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        math.exp(-1.0 * self.steps / self.epsilon_decay)
        self.steps += 1
        # If random number (0, 1) less than eps_threshold then we explore, else we exploit
        # In the long run, we will be exploring with P(epsilon_end)
        if random.random() < eps_threshold:
            # Explore: we choose a random action from the action space
            # You need to know your action space for this one
            # -------------------------------------------------------- #
            # Action space (n = 2)                                     #
	        # Moving = 0, 1 (false/true)                               #
            # Steering = -1, 0, 1 (turn left, no turn, turn right)     #
            # -------------------------------------------------------- #

            # Define the discrete action space
            action = np.random.choice(np.arange(0, 6))

            return torch.tensor([action], device=device, dtype=torch.long)
        else:
            # Exploit: we choose the best action at the state
            with torch.inference_mode(): 
                return self.policy_net(state).max(1).indices.view(1, 1)

    def decode_action(self, action):
        map = {
                0: [0, 0],
                1: [0, -1],
                2: [0, 1],
                3: [1, 0],
                4: [1, -1],
                5: [1, 1]
              }
        
        return map[action]

    def encode_action(self, action):
        action = tuple(action)
        map = {
                (0, 0): 0,
                (0, -1): 1,
                (0, 1): 2,
                (1, 0): 3,
                (1, -1): 4,
                (1, 1): 5
              }
        return map[action]
            
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
        action_value = self.policy_net(state_batch).gather(1, action_batch)

        # Q(s', a') for all next_states that are not None
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.inference_mode():
            # Double DQN -> Q_t(s', argmax Q_p(s', a'))
            # Currently standard DQN
            next_state_values[next_state_mask] = self.target_net(next_state_batch).max(1).values # gather(1, self.policy_net(next_state_batch).argmax(1, keepdim=True)).squeeze(1)
        
        # If the next_state is None, then the Q(s', a') term evaluates to 0
        expected_action_value = next_state_values * self.gamma + reward_batch

        # Huber loss
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(action_value, expected_action_value.unsqueeze(1))

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
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1 - self.tau)
        
        self.target_net.load_state_dict(target_net_state_dict)

    # We observe a state then we decide what action to take
    # and then we WAIT for the agent to ask us what action to take
    # we tell it what to take and WAIT for it to report back to us
    # what the next state and reward was
    def observe(self, data):
        # We just want to observe the state here, we don't really care about reward
        state = data
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        state_tensor = state_tensor.unsqueeze(0)
        action_tensor = self.select_action(state_tensor)
        action = action_tensor.tolist()

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
        torch.save(self.policy_net.state_dict(), f"save/{self.episodes_total}_Policy_RBX.pth")
        torch.save(self.target_net.state_dict(), f"save/{self.episodes_total}_Target_RBX.pth")
        torch.save(self.optimizer.state_dict(), f"save/{self.episodes_total}_Optim_RBX.pth")
    
    def get_stats(self):
        print(f"Steps taken: {self.steps}\nEpisodes trained: {self.episodes_total}")
