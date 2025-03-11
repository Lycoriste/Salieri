# State representation design/build
```python
import torch
import torch.nn as nn

class DeepQNetwork(nn.module):
    # Initialize NN object
    def __init__:
        super().__init__()
        # Still planing what goes here
```
> "A state in reinforcement learning is a representation of the current environment that the agent is in."

Essentially, it is anything we want it to be. This is a crucial part of reinforcement learning and arguably the hardest aspect in building an efficient and effective model.

It struggles from the curse of dimensionality, and faces the same problems we typically encounter with feature engineering in supervised learning.

A large part of the project is designing multiple prototypes of the state and see which works best. For no particular reason besides extensive documentation, this project will be using Deep Q Learning (DQN).