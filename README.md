# Salieri

Personal Roblox reinforcement learning tool and PyTorch wrapper for Roblox.

## Build Instructions for Dummies 
```bash
git clone https://github.com/Lycoriste/Salieri.git
cd Salieri
cmake -S . -B build
cmake --build build

# (Optional but recommended)
python3 -m venv .venv

# Linux/Mac
source .venv/bin/activate

# Windows
.venv/Scripts/activate

pip install -r requirements.txt
```

## Dependencies 
CMake

C++
- ASIO
- msgpack-c
- libtorch

Python
- torch
- matplotlib/plotly

Roblox
- msgpack-luau (ZarsBranchkin)

## Plans
- Session manager
- More reinforcement learning algorithms
- Replicate training environment (Python + Gym)
- LLMs
- Interpreter module
- Cybersecurity
- PyTorch -> libtorch

