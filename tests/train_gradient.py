"""
Training script optimized for cloud GPU training (Colab, Paperspace, etc).
This version is similar to train_camera.py but optimized for headless GPU environments.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from env.camera_line_follow_env import CameraLineFollowEnv
from config.config import TRAINING_CONFIG, LOGGING_CONFIG, HUGGINGFACE_CONFIG

def main():
    """Train the model with GPU acceleration"""
    # Use config settings
    config = TRAINING_CONFIG
    log_config = LOGGING_CONFIG
    
    # Create directories for outputs
    models_dir = Path("models")
    checkpoints_dir = models_dir / "checkpoints"
    best_model_dir = models_dir / "best_model"
    logs_dir = Path("logs")
    
    for directory in [models_dir, checkpoints_dir, best_model_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CarV1 GPU TRAINING")
    print("=" * 60)
    print(f"Total timesteps: {config['total_timesteps']:,}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Model save path: {models_dir}")
    print("=" * 60)
    
    # Create vectorized environment
    print("\nCreating training environment...")
    env = make_vec_env(
        CameraLineFollowEnv,
        n_envs=1,
        env_kwargs={"render_mode": None}  # No rendering on GPU
    )
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        CameraLineFollowEnv,
        n_envs=1,
        env_kwargs={"render_mode": None}
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=log_config["save_freq"],
        save_path=str(checkpoints_dir),
        name_prefix="ppo_camera_line_follow",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(logs_dir),
        eval_freq=log_config["eval_freq"],
        deterministic=True,
        render=False,
        n_eval_episodes=log_config["n_eval_episodes"],
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Optional: Add HuggingFace integration if enabled
    if HUGGINGFACE_CONFIG.get("enabled", False):
        try:
            from utils.huggingface import HuggingFaceCallback
            hf_callback = HuggingFaceCallback(
                repo_id=HUGGINGFACE_CONFIG["repo_id"],
                model_name="ppo_camera_line_follow",
                save_freq=HUGGINGFACE_CONFIG["push_freq"],
            )
            callbacks.callbacks.append(hf_callback)
            print(f"\n✓ HuggingFace integration enabled: {HUGGINGFACE_CONFIG['repo_id']}")
        except Exception as e:
            print(f"\n⚠ HuggingFace integration disabled: {e}")
    
    # Create the model with GPU support
    print("\nInitializing PPO model...")
    model = PPO(
        config["policy_type"],
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        verbose=1,
        tensorboard_log=str(logs_dir),
        device="auto",  # Will use CUDA if available
    )
    
    print(f"\nUsing device: {model.device}")
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")
    
    # Train the model
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            progress_bar=True,
        )
        
        # Save final model
        final_model_path = models_dir / "ppo_camera_line_follow_final"
        model.save(final_model_path)
        print(f"\n✓ Training complete! Final model saved to: {final_model_path}")
        
        # Push to HuggingFace if enabled
        if HUGGINGFACE_CONFIG.get("enabled", False):
            try:
                from utils.huggingface import push_final_model
                push_final_model(
                    model=model,
                    repo_id=HUGGINGFACE_CONFIG["repo_id"],
                    model_name="ppo_camera_line_follow_final",
                    env=env,
                )
                print(f"✓ Model pushed to HuggingFace: {HUGGINGFACE_CONFIG['repo_id']}")
            except Exception as e:
                print(f"⚠ Failed to push to HuggingFace: {e}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        interrupted_model_path = models_dir / "ppo_camera_line_follow_interrupted"
        model.save(interrupted_model_path)
        print(f"✓ Model saved to: {interrupted_model_path}")
    
    finally:
        # Clean up
        env.close()
        eval_env.close()
        print("\n" + "=" * 60)
        print("Training session ended")
        print("=" * 60)

if __name__ == "__main__":
    main()
