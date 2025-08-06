import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math, random
import os, sys, platform
from collections import defaultdict
from network import Multi_Head_QN
from replay_memory import ReplayMemory, Transition
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)