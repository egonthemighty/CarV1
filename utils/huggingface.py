"""
Hugging Face Hub integration utilities for model sharing and versioning.
"""

from stable_baselines3.common.callbacks import BaseCallback
from huggingface_hub import HfApi, create_repo, login
from pathlib import Path
import os


class HuggingFaceCallback(BaseCallback):
    """
    Callback for pushing model checkpoints to Hugging Face Hub during training.
    
    Args:
        repo_id: Hugging Face repository ID (e.g., 'username/carv1-model')
        push_freq: Push to hub every N training steps
        checkpoint_dir: Directory where checkpoints are saved
        private: Whether the repository should be private
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        repo_id: str,
        push_freq: int = 50000,
        checkpoint_dir: str = "./models/checkpoints",
        private: bool = False,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.repo_id = repo_id
        self.push_freq = push_freq
        self.checkpoint_dir = Path(checkpoint_dir)
        self.private = private
        self.api = HfApi()
        self.repo_created = False
        
    def _init_callback(self) -> None:
        """Initialize callback - create repo if needed."""
        if not self.repo_created:
            try:
                # Try to create repo (will skip if already exists)
                create_repo(
                    repo_id=self.repo_id,
                    repo_type="model",
                    private=self.private,
                    exist_ok=True
                )
                self.repo_created = True
                if self.verbose > 0:
                    print(f"✓ Connected to Hugging Face repo: {self.repo_id}")
            except Exception as e:
                if self.verbose > 0:
                    print(f"⚠ Warning: Could not create HF repo: {e}")
                    print("  Make sure you're logged in: huggingface-cli login")
    
    def _on_step(self) -> bool:
        """Called after each training step."""
        if self.n_calls % self.push_freq == 0:
            self._push_to_hub()
        return True
    
    def _push_to_hub(self) -> None:
        """Push current checkpoint to Hugging Face Hub."""
        try:
            if self.verbose > 0:
                print(f"\n📤 Pushing checkpoint to Hugging Face Hub (step {self.n_calls})...")
            
            # Upload checkpoint directory
            self.api.upload_folder(
                folder_path=str(self.checkpoint_dir),
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"Update checkpoint at step {self.n_calls}"
            )
            
            # Upload tensorboard logs if available
            tensorboard_dir = Path("./debug_output/tensorboard")
            if tensorboard_dir.exists():
                self.api.upload_folder(
                    folder_path=str(tensorboard_dir),
                    path_in_repo="tensorboard",
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=f"Update tensorboard logs at step {self.n_calls}"
                )
            
            if self.verbose > 0:
                print(f"✓ Successfully pushed to {self.repo_id}")
                
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠ Warning: Failed to push to HF Hub: {e}")


def push_final_model(
    model,
    repo_id: str,
    model_name: str = "final_model",
    private: bool = False,
    commit_message: str = "Upload final trained model"
):
    """
    Push the final trained model to Hugging Face Hub.
    
    Args:
        model: Trained Stable Baselines3 model
        repo_id: Hugging Face repository ID
        model_name: Name for the model file
        private: Whether the repository should be private
        commit_message: Commit message for the upload
    """
    try:
        # Save model locally first
        model_path = Path("./models") / model_name
        model.save(str(model_path))
        
        # Create repo if needed
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        
        # Upload the model
        api = HfApi()
        api.upload_file(
            path_or_fileobj=f"{model_path}.zip",
            path_in_repo=f"{model_name}.zip",
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message
        )
        
        print(f"✓ Final model uploaded to https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"⚠ Error uploading final model: {e}")


def create_model_card(
    repo_id: str,
    model_info: dict,
    training_info: dict,
    evaluation_info: dict = None
):
    """
    Create a model card (README.md) for the Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID
        model_info: Dictionary with model details (algorithm, architecture, etc.)
        training_info: Dictionary with training details (timesteps, hyperparameters, etc.)
        evaluation_info: Optional dictionary with evaluation metrics
    """
    
    card_content = f"""---
library_name: stable-baselines3
tags:
- CarV1-v0
- deep-reinforcement-learning
- reinforcement-learning
- stable-baselines3
model-index:
- name: {model_info.get('name', 'CarV1')}
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: CarV1-v0
      type: CarV1-v0
    metrics:
    - type: mean_reward
      value: {evaluation_info.get('mean_reward', 'N/A') if evaluation_info else 'N/A'}
      name: mean_reward
---

# **{model_info.get('name', 'CarV1 Self-Driving Agent')}**

This is a trained model for the CarV1 custom Gymnasium environment using **{model_info.get('algorithm', 'PPO')}** algorithm.

## Environment

Custom self-driving car environment with:
- **Observation Space**: Car position, velocity, heading, and 8 distance sensors
- **Action Space**: Continuous steering and throttle control
- **Goal**: Learn to navigate without collisions while maximizing speed

## Training Details

- **Algorithm**: {model_info.get('algorithm', 'PPO')}
- **Total Timesteps**: {training_info.get('total_timesteps', 'N/A'):,}
- **Learning Rate**: {training_info.get('learning_rate', 'N/A')}
- **Batch Size**: {training_info.get('batch_size', 'N/A')}
- **Training Time**: {training_info.get('training_time', 'N/A')}

### Hyperparameters
```python
{training_info.get('hyperparameters', 'See training script')}
```

## Evaluation
"""
    
    if evaluation_info:
        card_content += f"""
- **Mean Reward**: {evaluation_info.get('mean_reward', 'N/A')}
- **Std Reward**: {evaluation_info.get('std_reward', 'N/A')}
- **Mean Episode Length**: {evaluation_info.get('mean_episode_length', 'N/A')}
- **Success Rate**: {evaluation_info.get('success_rate', 'N/A')}
"""
    else:
        card_content += "\nEvaluation metrics will be added after training.\n"
    
    card_content += """
## Usage

```python
from stable_baselines3 import PPO
from env.car_env import CarEnv

# Load the model
model = PPO.load("path/to/model")

# Create environment
env = CarEnv(render_mode="human")

# Run the model
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

## Model Architecture

Policy network: Multi-layer perceptron (MLP)
- Input: 14 dimensions (position, velocity, angle, sensors)
- Output: 2 dimensions (steering, throttle)

## Repository

Project repository: [CarV1 on GitHub](https://github.com/yourusername/CarV1)
"""
    
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card"
        )
        print(f"✓ Model card created for {repo_id}")
    except Exception as e:
        print(f"⚠ Error creating model card: {e}")


def load_from_hub(repo_id: str, filename: str = "final_model"):
    """
    Load a model from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID
        filename: Name of the model file (without .zip extension)
    
    Returns:
        Loaded model
    """
    from huggingface_hub import hf_hub_download
    from stable_baselines3 import PPO
    
    try:
        # Download model
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{filename}.zip",
            repo_type="model"
        )
        
        # Load model
        model = PPO.load(model_path[:-4])  # Remove .zip extension
        print(f"✓ Model loaded from {repo_id}")
        return model
        
    except Exception as e:
        print(f"⚠ Error loading model from hub: {e}")
        return None
