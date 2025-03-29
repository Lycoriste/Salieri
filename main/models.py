import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# NOTE - Barebone structure for DQN
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # Convolutional neural networks perform well for spatial navigation
        self.sequence = nn.Sequential(
                   nn.Conv2D(in_channels=1, out_channels=1, kernel_size=8, stride=4),
                   nn.Relu(),

                ) 

    def forward(self, x):
        return self.layer3(x)

class Sample_DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # Channel, height, and width
        channel, _, _ = state_dim

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        # Normalize x -> (0, 255)
        x /= 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
