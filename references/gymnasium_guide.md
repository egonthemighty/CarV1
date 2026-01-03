# Custom Gymnasium Environment Guide

## Creating a Custom Gymnasium Environment

### Basic Structure

All custom environments must inherit from `gym.Env` and implement these methods:

```python
import gymnasium as gym
from gymnasium import spaces

class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(...)
        self.observation_space = spaces.Box(...)
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        # Return observation, info
        return observation, info
    
    def step(self, action):
        """Execute one time step"""
        # Return observation, reward, terminated, truncated, info
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        pass
    
    def close(self):
        """Clean up resources"""
        pass
```

### Space Types

#### Box Space (Continuous)
```python
# Continuous values in a range
spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
```

#### Discrete Space
```python
# Integer values from 0 to n-1
spaces.Discrete(n)
```

#### MultiDiscrete Space
```python
# Multiple discrete values
spaces.MultiDiscrete([3, 4, 5])  # 3 actions, 4 actions, 5 actions
```

### Observation Design

Good observations should:
1. Be normalized (typically [-1, 1] or [0, 1])
2. Include relevant state information
3. Not include redundant information
4. Be consistent in scale across dimensions

Example for car:
- Position (normalized to environment size)
- Velocity (normalized to max speed)
- Angle (radians)
- Sensor readings (normalized to max range)

### Reward Shaping

Principles:
1. **Sparse vs Dense**: Dense rewards (every step) learn faster than sparse (only at goal)
2. **Scale**: Keep rewards roughly in range [-1, 1] per step
3. **Components**: Combine multiple reward signals
4. **Avoid Exploitation**: Test for unintended optimal behaviors

Example reward structure:
```python
reward = 0.0
reward += progress_toward_goal * 0.1      # Encourage progress
reward -= distance_from_center * 0.01     # Stay on track
reward -= abs(action_change) * 0.001      # Smooth actions
if collision:
    reward -= 10.0                         # Strong penalty
if reached_goal:
    reward += 100.0                        # Strong reward
```

### Termination Conditions

**Terminated**: Episode ends due to task completion or failure
```python
terminated = (reached_goal or collision or out_of_bounds)
```

**Truncated**: Episode ends due to time limit
```python
truncated = (steps >= max_steps)
```

### Rendering

Modes:
- `"human"`: Display to screen
- `"rgb_array"`: Return numpy array
- `None`: No rendering (fastest for training)

```python
def __init__(self, render_mode=None):
    self.render_mode = render_mode
    
def render(self):
    if self.render_mode == "human":
        # Display to screen
        pass
    elif self.render_mode == "rgb_array":
        # Return RGB array
        return np.array(...)
```

### Registration

Register environment for easy creation:
```python
# In __init__.py
from gymnasium.envs.registration import register

register(
    id='CustomEnv-v0',
    entry_point='module.path:CustomEnv',
    max_episode_steps=1000,
)

# Usage
import gymnasium as gym
env = gym.make('CustomEnv-v0')
```

## Best Practices

### 1. Vectorized Environments
Use vectorized environments for faster training:
```python
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Single process
env = DummyVecEnv([lambda: YourEnv() for _ in range(4)])

# Multi-process (faster)
env = SubprocVecEnv([lambda: YourEnv() for _ in range(4)])
```

### 2. Monitoring
Wrap environment with Monitor to track performance:
```python
from stable_baselines3.common.monitor import Monitor

env = Monitor(env, log_dir)
```

### 3. Normalization
Normalize observations and rewards:
```python
from stable_baselines3.common.vec_env import VecNormalize

env = VecNormalize(env, norm_obs=True, norm_reward=True)
```

### 4. Testing
Always test your environment:
```python
from gymnasium.utils.env_checker import check_env

check_env(env)  # Validates environment implementation
```

### 5. Determinism
Support seeding for reproducibility:
```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    if seed is not None:
        np.random.seed(seed)
    # ... reset logic
```

## Common Pitfalls

### 1. Incorrect Space Bounds
```python
# BAD: Unbounded when it should be bounded
self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))

# GOOD: Realistic bounds
self.observation_space = spaces.Box(low=-10, high=10, shape=(4,))
```

### 2. State Not Reset Properly
```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    
    # BAD: Forgetting to reset all state variables
    self.position = [0, 0]
    # self.velocity not reset!
    
    # GOOD: Reset all state
    self.position = [0, 0]
    self.velocity = [0, 0]
    self.angle = 0
```

### 3. Reward Scaling Issues
```python
# BAD: Rewards vary wildly in scale
reward = distance * 1000 - time * 0.001 + goal_bonus * 10000

# GOOD: Similar scales
reward = distance * 0.1 - time * 0.1 + goal_bonus * 1.0
```

### 4. Forgetting to Return Proper Tuple
```python
# BAD
def step(self, action):
    return observation, reward

# GOOD
def step(self, action):
    return observation, reward, terminated, truncated, info
```

## Debugging Tips

1. **Print observations**: Check if they're in expected range
2. **Visualize**: Use render mode to see what's happening
3. **Random agent**: Test with random actions first
4. **Check spaces**: Use `env.observation_space.sample()` and `env.action_space.sample()`
5. **Log everything**: Track positions, velocities, rewards over time
6. **Start simple**: Begin with simple reward, add complexity gradually
