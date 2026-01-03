# Hugging Face Hub Integration Guide

## Setup

### 1. Install Dependencies
Already included in `requirements.txt`:
```bash
pip install huggingface-hub
```

### 2. Login to Hugging Face
```bash
huggingface-cli login
```
Or set your token:
```python
from huggingface_hub import login
login(token="your_token_here")
```

Get your token from: https://huggingface.co/settings/tokens

### 3. Configure Training

Edit `config/config.py`:
```python
HUGGINGFACE_CONFIG = {
    'enabled': True,  # Enable HF integration
    'repo_id': 'your-username/carv1-model',  # Your HF repo
    'private': False,  # Public or private repo
    'push_freq': 50000,  # Push every 50k steps
    'push_on_complete': True,  # Push final model
    'include_tensorboard': True,  # Include tensorboard logs
}
```

## Features

### Automatic Checkpoint Pushing
During training, checkpoints are automatically pushed to HuggingFace Hub at the specified frequency. This provides:
- Version control for your models
- Backup during training
- Easy sharing and collaboration
- Training progress tracking

### Model Card Generation
Automatically creates a README.md with:
- Model description
- Training hyperparameters
- Evaluation metrics
- Usage examples
- Architecture details

### Final Model Upload
After training completes:
- Final model is uploaded
- Model card is updated with results
- Tensorboard logs are included

## Usage Examples

### Training with HF Integration
```python
# Simply run your training script with HF enabled in config
python tests/train.py
```

The script will:
1. Create HF repository if it doesn't exist
2. Push checkpoints every `push_freq` steps
3. Upload final model when training completes
4. Generate model card with training info

### Loading Models from Hub
```python
from utils.huggingface import load_from_hub
from env.car_env import CarEnv

# Load trained model
model = load_from_hub("your-username/carv1-model", "final_model")

# Use it
env = CarEnv(render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### Manual Push
```python
from utils.huggingface import push_final_model, create_model_card

# Push a specific model
push_final_model(
    model=model,
    repo_id="your-username/carv1-model",
    model_name="experiment_v1",
    commit_message="Experiment with higher learning rate"
)

# Create/update model card
model_info = {
    'name': 'CarV1 Experiment V1',
    'algorithm': 'PPO',
}

training_info = {
    'total_timesteps': 1_000_000,
    'learning_rate': 5e-4,
    'batch_size': 128,
    'training_time': '2h 30m',
}

evaluation_info = {
    'mean_reward': 450.2,
    'std_reward': 125.3,
    'mean_episode_length': 850,
}

create_model_card(
    repo_id="your-username/carv1-model",
    model_info=model_info,
    training_info=training_info,
    evaluation_info=evaluation_info
)
```

## Repository Structure on HF Hub

Your repository will contain:
```
your-username/carv1-model/
├── README.md                    # Model card
├── carv1_final_model.zip        # Trained model
├── carv1_model_10000_steps.zip  # Checkpoint
├── carv1_model_20000_steps.zip  # Checkpoint
├── ...
└── tensorboard/                 # Training logs
    └── PPO_1/
        └── events.out.tfevents...
```

## Best Practices

### 1. Use Meaningful Commit Messages
```python
commit_message=f"Training checkpoint - Reward: {mean_reward:.2f}, Step: {step}"
```

### 2. Version Your Experiments
Use different repo names or branches for different experiments:
- `username/carv1-baseline`
- `username/carv1-curriculum`
- `username/carv1-dense-rewards`

### 3. Include Evaluation Metrics
Update model card with evaluation results:
```python
evaluation_info = {
    'mean_reward': evaluate_policy(model, env, n_eval_episodes=100)[0],
    'success_rate': calculate_success_rate(model, env),
}
```

### 4. Keep Models Organized
Use descriptive filenames:
- `carv1_final_model` - Final trained model
- `carv1_best_eval` - Best performing on evaluation
- `carv1_checkpoint_500k` - Specific checkpoint

### 5. Private vs Public
- Use `private=True` for experiments in progress
- Make `private=False` when ready to share
- You can change this later on HF website

## Troubleshooting

### Authentication Error
```
Error: Invalid token
```
**Solution**: Run `huggingface-cli login` and enter your token

### Repository Not Found
```
Error: Repository not found
```
**Solution**: Check `repo_id` format is `username/repo-name` and you have permissions

### Upload Failed
```
Error: Failed to push to HF Hub
```
**Solutions**:
- Check internet connection
- Verify repo permissions
- Check file sizes (HF has limits on individual files)
- Ensure you're logged in

### Slow Uploads
If uploads are taking too long:
- Increase `push_freq` to push less often
- Disable tensorboard uploads: `include_tensorboard: False`
- Use compression for large checkpoints

## Advanced Features

### Custom Callbacks
Create your own HF callback for specific needs:
```python
from utils.huggingface import HuggingFaceCallback

class CustomHFCallback(HuggingFaceCallback):
    def _on_step(self):
        # Custom logic before push
        if self.should_push():
            self.add_metadata()
            self._push_to_hub()
        return True
```

### Downloading Specific Checkpoints
```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id="your-username/carv1-model",
    filename="carv1_model_50000_steps.zip",
    repo_type="model"
)
```

### Comparing Models
```python
# Load different versions
model_v1 = load_from_hub("username/carv1-model", "carv1_v1")
model_v2 = load_from_hub("username/carv1-model", "carv1_v2")

# Evaluate both
results_v1 = evaluate_policy(model_v1, env, n_eval_episodes=100)
results_v2 = evaluate_policy(model_v2, env, n_eval_episodes=100)

print(f"V1: {results_v1[0]:.2f} ± {results_v1[1]:.2f}")
print(f"V2: {results_v2[0]:.2f} ± {results_v2[1]:.2f}")
```

## Resources

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)
- [huggingface_hub Python Library](https://huggingface.co/docs/huggingface_hub/index)
- [Model Cards Guide](https://huggingface.co/docs/hub/model-cards)
- [Stable Baselines3 + HF](https://stable-baselines3.readthedocs.io/en/master/guide/huggingface.html)
