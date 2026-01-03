# CarV1 - Self-Driving Car Gymnasium Environment

A custom Gymnasium environment for training reinforcement learning agents to drive a simulated car.

## Project Structure

```
CarV1/
├── env/                    # Custom gymnasium environment
├── tests/                  # Test scripts
├── debug_output/           # Debug outputs (logs, visualizations)
├── references/             # Documentation and reference materials
├── models/                 # Trained models and checkpoints
├── config/                 # Configuration files
└── utils/                  # Utility functions
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Important: Always Use the venv

If pip is installing to the wrong location, use the venv's python directly:
```bash
# Windows
.\venv\Scripts\python.exe -m pip install <package>
.\venv\Scripts\python.exe <script.py>

# Linux/Mac
./venv/bin/python -m pip install <package>
./venv/bin/python <script.py>
```

## Usage

```python
import gymnasium as gym
from env.car_env import CarEnv

# Create environment
env = CarEnv()

# Reset environment
observation, info = env.reset()

# Run episode
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## Development

- Run tests: `python -m pytest tests/`
- Check environment: `python tests/test_env.py`

## Model Repository

Trained models: [https://huggingface.co/egonthemighty/carV1](https://huggingface.co/egonthemighty/carV1)
