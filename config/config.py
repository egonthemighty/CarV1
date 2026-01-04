"""
Configuration settings for the CarV1 environment and training.

OPTIMIZATION NOTES:
- total_timesteps: Use 10k for testing, 500k+ for production
- n_steps & batch_size: Increased from defaults for better GPU utilization
- Environment renders directly at target resolution to avoid cv2.resize overhead
- Mixed precision training is automatic with PyTorch 1.6+ on CUDA GPUs
- Monitor time/fps in TensorBoard for performance validation
"""

# Environment Configuration
ENV_CONFIG = {
    'window_width': 800,
    'window_height': 600,
    'max_steps': 1000,
    'render_mode': None,  # 'human', 'rgb_array', or None
    'track_complexity': 1,
}

# Car Physics Configuration
CAR_CONFIG = {
    'length': 40,
    'width': 20,
    'max_speed': 100,
    'max_acceleration': 50,
    'friction': 0.95,
    'max_steering_angle': 45,  # degrees
}

# Sensor Configuration
SENSOR_CONFIG = {
    'num_sensors': 8,
    'sensor_range': 200,
    'sensor_angles': None,  # None for evenly distributed
}

# Reward Configuration
REWARD_CONFIG = {
    'speed_reward_factor': 0.01,
    'collision_penalty': -100,
    'wall_proximity_penalty_factor': 0.01,
    'wall_proximity_threshold': 50,
    'time_penalty': -0.1,
}

# Training Configuration
TRAINING_CONFIG = {
    'algorithm': 'PPO',
    'policy_type': 'CnnPolicy',  # CNN for raw pixels, MLP for features
    'total_timesteps': 10_000,  # Very short for testing (use 500k+ for real training)
    'learning_rate': 3e-4,
    'n_steps': 4096,  # Increased from 2048 for more efficient gradient updates
    'batch_size': 128,  # Increased from 64 for better GPU utilization
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_dir': './debug_output',
    'tensorboard_log': './debug_output/tensorboard',
    'save_freq': 10000,
    'checkpoint_dir': './models/checkpoints',
    'eval_freq': 5000,
    'n_eval_episodes': 10,
}

# Debug Configuration
DEBUG_CONFIG = {
    'enable_debug_output': True,
    'save_episode_videos': False,
    'save_trajectory_plots': True,
    'verbose': 1,
}

# Hugging Face Configuration
HUGGINGFACE_CONFIG = {
    'enabled': False,  # Set to True to enable HF integration
    'repo_id': 'egonthemighty/carV1',  # HuggingFace repository
    'private': False,  # Whether the repo should be private
    'push_freq': 50000,  # Push to HF every N steps
    'push_on_complete': True,  # Push final model when training completes
    'include_tensorboard': True,  # Include tensorboard logs
}
