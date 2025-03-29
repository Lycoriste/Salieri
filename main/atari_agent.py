import gymnasium as gym
from gymnasium.wrappers import RecordVideo as Monitor, ResizeObservation, GrayscaleObservation
import ale_py
import itertools
import numpy as np
import os, sys
import math, random
import matplotlib.pyplot as plt

from lib import plotting
from collections import deque, namedtuple, defaultdict
from IPython import display

# NOTE: Recreation of Atari agent from Mnih et al. paper
class ATARI_AGENT:
    def __init__(self, batch_size = 128, gamma = 0.99, epsilon_start = 0.99, epsilon_end = 0.01, epsilon_decay = 1000, tau = 0.005, alpha = 1e-4):
        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.alpha = alpha

        # Utilities
        self.memory = ReplayMemory(batch_size * 1000)
        self.steps = 0
        self.episode_reward = defaultdict(int)
        self.episode_length = defaultdict(int)
        self.episodes_total = 0

        # This is a continuous observation space (env.observation_space.n for Discrete)
        # One neural network for optimal policy and the other for behavior/exploration
        state_space = env.observation_space.shape
        state_space = (state_space[2], state_space[0], state_space[1])
        self.policy_net = DQN(state_space, env.action_space.n).to(device)
        self.target_net = DQN(state_space, env.action_space.n).to(device)
        # Apply policy_net parameters (weights and biases) to target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Adam - SGD either one works
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.alpha)

    # Epsilon greedy action selection
    def select_action(self, state):
        # Epsilon Threshold: Early in training -> prioritize exploring | Later in training -> exploit more
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        math.exp(-1.0 * self.steps / self.epsilon_decay)
        self.steps += 1

        # If random number (0, 1) less than eps_threshold then we explore, else we exploit
        if random.random() < eps_threshold:
            # Explore: we choose a random action
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        else:
            # Exploit: we choose the best action at the state
            with torch.inference_mode(): 
                return self.policy_net(state).max(1).indices.view(1, 1)
            
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
            next_state_values[next_state_mask] = self.target_net(next_state_batch).max(1).values #gather(1, self.policy_net(next_state_batch).argmax(1, keepdim=True)).squeeze(1)
        
        # If the next_state is None, then the Q(s', a') term evaluates to 0
        expected_action_value = next_state_values * self.gamma + reward_batch

        # Loss function for DQN -> L(s) = MSE
        # We use the Huber loss here as it is less sensitive to outliers
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(action_value, expected_action_value.unsqueeze(1))


        # Clear all gradients
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping - prevents them from exploding
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100.0)
        self.optimizer.step()

        # Soft updates
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
    
    # Train agent - for any actual improvements we need more than 128 episodes as one batch = 128
    def train(self, episodes: int = 128):
        plt.ion()
        fig, ax = plt.subplots()
        for ep in range(episodes):
            state_raw, _ = env.reset()

            while True:
                state_tensor = torch.tensor(state_raw, dtype=torch.float32, device=device)
                state_tensor = state_tensor.permute(2, 0 ,1).unsqueeze(0)
                action_tensor = self.select_action(state_tensor)
                action = action_tensor.item()

                next_state_raw, reward, done, truncated, _ = env.step(action)
                next_state_tensor = torch.tensor(next_state_raw, dtype=torch.float32, device=device) 
                next_state_tensor = next_state_tensor.permute(2, 0 ,1).unsqueeze(0)
                reward_tensor = torch.tensor([reward], device=device)

                self.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)

                self.episode_reward[self.episodes_total] += reward
                self.episode_length[self.episodes_total] += 1
                self.steps += 1

                self.optimize_model()

                if done or truncated:
                    break
                
                state_raw = next_state_raw

            # Update the plot every 10 episodes once training starts
            if (ep % 5 == 0):
                ax.clear()  # Clears the old plot
                ax.plot(list(self.episode_length.keys()), list(self.episode_length.values()), 'b-')  # Plots the updated data
                ax.set_xlabel("Episodes")
                ax.set_ylabel("Episode Length")
                ax.set_title("Episode Length Over Time")
                display.clear_output(wait=True)
                display.display(fig)
                
                print("Current step: ", self.steps)
                sys.stdout.flush()

            self.episodes_total += 1

        return self.episodes_total, self.episode_length, self.episode_reward
    
    def get_models(self):
        return self.policy_net, self.target_net