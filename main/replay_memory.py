import numpy as np
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Sumtree PER by Schaul et al.
class PriorityReplay:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=1e-4, eps=1e-5):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # [0~1] convert priority to power
        self.beta = beta    # importance sampling bias correction
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.eps = eps      # small constant to avoid zero priority

    def push(self, *args):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:]) if self.tree.n_entries > 0 else 1.0
        self.tree.add(max_priority, Transition(*args))

    def sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        sampling_probabilities = priorities / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= np.max(is_weights)

        return batch, idxs, is_weights

    def update_priorities(self, idxs, td_errors):
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + self.eps) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)        # sum tree
        self.data = np.zeros(capacity, dtype=object)  # exp buffer
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]