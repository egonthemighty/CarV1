# CarV1 Project Overview

## Purpose
This is a custom Gymnasium environment for training reinforcement learning agents to drive a simulated car. The project uses PPO (Proximal Policy Optimization) from Stable Baselines3 to train self-driving behavior.

## Project Structure

### Core Components

#### `/env/` - Custom Gymnasium Environment
- `car_env.py`: Main environment implementation
  - Observation space: car position, velocity, angle, angular velocity, and 8 distance sensors
  - Action space: steering [-1, 1] and throttle [-1, 1]
  - Physics simulation with friction, acceleration, and collision detection
  - Pygame rendering for visualization

#### `/config/` - Configuration
- `config.py`: All configuration parameters
  - Environment settings (window size, max steps)
  - Car physics parameters (speed, acceleration, steering)
  - Sensor configuration (8 sensors, 200 unit range)
  - Training hyperparameters (PPO settings)
  - Reward shaping parameters

#### `/utils/` - Utilities
- `helpers.py`: Helper functions
  - Plotting trajectories and rewards
  - Saving episode data
  - Logger creation

#### `/tests/` - Test and Training Scripts
- `test_env.py`: Environment validation suite
- `visualize_env.py`: Visual testing with random actions
- `train.py`: Main training script with PPO

### Supporting Directories
- `/debug_output/`: Logs, plots, and debug visualizations
- `/models/checkpoints/`: Model checkpoints during training
- `/references/`: Documentation (this file)

## Key Implementation Details

### Observation Space (14 dimensions)
1. X position (0 to window_width)
2. Y position (0 to window_height)
3. X velocity (-max_speed to max_speed)
4. Y velocity (-max_speed to max_speed)
5. Heading angle (-π to π)
6. Angular velocity (-π to π)
7-14. Eight distance sensors (0 to sensor_range)

### Action Space (2 dimensions)
1. Steering: -1 (left) to +1 (right)
2. Throttle: -1 (brake) to +1 (accelerate)

### Reward Function
- Positive reward for forward speed (0.01 * speed)
- Penalty for proximity to walls
- Large penalty for collisions (-100)
- Small time penalty to encourage efficiency (-0.1)

### Physics Model
- Velocity updated based on throttle and current angle
- Friction applied each step (0.95 factor)
- Angular velocity influenced by steering and speed
- Position updated from velocity

## Usage Examples

### Test Environment
```python
python tests/test_env.py
```

### Visualize Random Agent
```python
python tests/visualize_env.py
```

### Train Agent
```python
python tests/train.py
```

### Monitor Training
```bash
tensorboard --logdir debug_output/tensorboard
```

## Next Steps & Improvements

### Short Term
- Add more complex track layouts (walls, obstacles)
- Implement waypoint-based rewards
- Add configurable difficulty levels
- Create evaluation metrics dashboard

### Medium Term
- Add multiple car types with different physics
- Implement multi-agent scenarios
- Add realistic friction/tire model
- Create procedurally generated tracks

### Long Term
- 3D rendering with better graphics
- Integration with real sensor data
- Transfer learning to real vehicles
- Multi-objective optimization (speed vs safety)

## Links

- **HuggingFace Models**: [https://huggingface.co/egonthemighty/carV1](https://huggingface.co/egonthemighty/carV1)
- **Gymnasium Documentation**: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)

## Common Issues & Solutions

### Import Errors
- Make sure to add project root to Python path
- Install all requirements: `pip install -r requirements.txt`

### Pygame Window Not Showing
- Check render_mode is set to "human"
- Ensure pygame is properly installed

### Training Not Converging
- Adjust reward function in config/config.py
- Modify PPO hyperparameters (learning rate, batch size)
- Check if car is getting stuck immediately (collision penalty too high)

### Performance Issues
- Set render_mode to None during training
- Reduce sensor count
- Lower environment resolution
