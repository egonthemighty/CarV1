# Reinforcement Learning Training Guide

## Overview of Reinforcement Learning

### Key Concepts

**Agent**: The learner/decision maker (our car)
**Environment**: The world the agent interacts with
**State (s)**: Current situation of the environment
**Action (a)**: Choice made by the agent
**Reward (r)**: Feedback from environment
**Policy (π)**: Strategy for choosing actions

### The RL Loop
```
1. Agent observes state s
2. Agent takes action a based on policy π
3. Environment transitions to new state s'
4. Environment provides reward r
5. Agent updates policy
6. Repeat
```

## PPO (Proximal Policy Optimization)

### Why PPO?

PPO is a popular choice because:
- **Stable**: Doesn't make huge policy changes
- **Sample Efficient**: Works well with limited data
- **Versatile**: Works on continuous and discrete actions
- **Easy to Tune**: Fewer hyperparameters than alternatives

### How PPO Works

1. **Collect Experience**: Run current policy for N steps
2. **Estimate Advantages**: Calculate how good actions were
3. **Update Policy**: Improve policy, but not too much
4. **Repeat**: Continue until convergence

### Key Hyperparameters

#### Learning Rate (`learning_rate`)
- **Range**: 1e-5 to 1e-3
- **Default**: 3e-4
- **Effect**: How big policy updates are
- **Tune**: Decrease if unstable, increase if too slow

#### Number of Steps (`n_steps`)
- **Range**: 128 to 4096
- **Default**: 2048
- **Effect**: How much experience before update
- **Tune**: Increase for complex tasks, decrease for simple

#### Batch Size (`batch_size`)
- **Range**: 32 to 512
- **Default**: 64
- **Effect**: How many samples per gradient update
- **Tune**: Larger = more stable but slower

#### Number of Epochs (`n_epochs`)
- **Range**: 3 to 30
- **Default**: 10
- **Effect**: How many times to use collected data
- **Tune**: Increase if data collection is expensive

#### Discount Factor (`gamma`)
- **Range**: 0.9 to 0.999
- **Default**: 0.99
- **Effect**: How much to value future rewards
- **Tune**: Higher for long-term tasks, lower for immediate

#### GAE Lambda (`gae_lambda`)
- **Range**: 0.9 to 0.99
- **Default**: 0.95
- **Effect**: Bias-variance tradeoff in advantage estimation
- **Tune**: Rarely needs changing

#### Clip Range (`clip_range`)
- **Range**: 0.1 to 0.3
- **Default**: 0.2
- **Effect**: How much policy can change per update
- **Tune**: Decrease for more stable but slower learning

#### Entropy Coefficient (`ent_coef`)
- **Range**: 0.0 to 0.1
- **Default**: 0.01
- **Effect**: Encourages exploration
- **Tune**: Increase if stuck in local optimum

## Training Pipeline

### 1. Environment Setup
```python
from env.car_env import CarEnv
from stable_baselines3.common.monitor import Monitor

env = CarEnv(render_mode=None)
env = Monitor(env, log_dir)
```

### 2. Model Creation
```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",           # Multi-layer perceptron
    env,
    learning_rate=3e-4,
    verbose=1,
    tensorboard_log="./logs"
)
```

### 3. Callbacks
```python
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Save checkpoints
checkpoint = CheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints",
    name_prefix="model"
)

# Evaluate periodically
eval_callback = EvalCallback(
    eval_env,
    eval_freq=5000,
    best_model_save_path="./best_model"
)
```

### 4. Training
```python
model.learn(
    total_timesteps=1_000_000,
    callback=[checkpoint, eval_callback]
)
```

### 5. Saving and Loading
```python
# Save
model.save("final_model")

# Load
model = PPO.load("final_model")
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./debug_output/tensorboard
```

Key metrics to watch:
- **ep_rew_mean**: Average episode reward (should increase)
- **ep_len_mean**: Average episode length
- **loss**: Training loss (should decrease)
- **policy_gradient_loss**: Policy network loss
- **value_loss**: Value network loss
- **entropy_loss**: Exploration bonus
- **learning_rate**: Current learning rate

### Evaluating Progress

Good signs:
- Episode reward increasing
- Episode length increasing (if longer is better)
- Consistent improvement over time

Bad signs:
- Reward plateauing early
- High variance in rewards
- Loss exploding or oscillating

## Troubleshooting

### Not Learning (Flat Reward)

**Possible causes:**
1. Reward scale too small/large
2. Learning rate too high/low
3. Initial random policy doesn't find any reward
4. Environment too hard

**Solutions:**
- Check reward values (should be roughly -1 to 1 per step)
- Try smaller/larger learning rate
- Curriculum learning: start easy, make harder
- Add reward shaping to guide agent

### Unstable Learning (Reward Oscillates)

**Possible causes:**
1. Learning rate too high
2. Batch size too small
3. Updates too aggressive

**Solutions:**
- Decrease learning rate
- Increase batch size
- Decrease clip_range
- Increase n_steps

### Learning Then Forgetting (Catastrophic Forgetting)

**Possible causes:**
1. Clip range too high
2. Too many epochs
3. Learning rate too high

**Solutions:**
- Decrease clip_range
- Decrease n_epochs
- Decrease learning_rate
- Use replay buffer (if switching to SAC/TD3)

### Sample Inefficient (Too Slow)

**Possible causes:**
1. Sparse rewards
2. Large state/action space
3. Inefficient exploration

**Solutions:**
- Add reward shaping
- Increase entropy coefficient
- Use multiple parallel environments
- Consider different algorithm (SAC, TD3)

## Advanced Techniques

### Curriculum Learning
Start with easy tasks, gradually increase difficulty:
```python
# Start with simple track
env = CarEnv(track_complexity=1)
model.learn(100_000)

# Increase complexity
env = CarEnv(track_complexity=2)
model.set_env(env)
model.learn(100_000)
```

### Reward Shaping
Guide learning with intermediate rewards:
```python
# Instead of only rewarding goal
reward = 1.0 if reached_goal else 0.0

# Reward progress
reward = distance_to_goal_decreased * 0.1
reward += speed_in_right_direction * 0.01
reward += 1.0 if reached_goal else 0.0
```

### Transfer Learning
Use pre-trained model as starting point:
```python
# Load pre-trained model
model = PPO.load("pretrained_model")

# Continue training on new task
model.set_env(new_env)
model.learn(additional_timesteps)
```

### Hyperparameter Tuning
Use Optuna for automatic tuning:
```python
import optuna
from stable_baselines3 import PPO

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_int("n_steps", 128, 4096)
    
    model = PPO("MlpPolicy", env, learning_rate=lr, n_steps=n_steps)
    model.learn(100_000)
    
    # Evaluate
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10)
    return mean_reward

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

## Evaluation

### Testing Trained Model
```python
model = PPO.load("best_model")
env = CarEnv(render_mode="human")

obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### Computing Metrics
```python
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(
    model, 
    env, 
    n_eval_episodes=100,
    deterministic=True
)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
```

## Resources

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
