"""
Training script for the CarV1 self-driving car agent.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from env.car_env import CarEnv
from config.config import TRAINING_CONFIG, LOGGING_CONFIG, HUGGINGFACE_CONFIG
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from utils.huggingface import HuggingFaceCallback, push_final_model, create_model_card
import os
import time


def main():
    """Train the agent."""
    print("Initializing training...")
    
    # Start timing
    start_time = time.time()
    
    # Create directories
    os.makedirs(LOGGING_CONFIG['log_dir'], exist_ok=True)
    os.makedirs(LOGGING_CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(LOGGING_CONFIG['tensorboard_log'], exist_ok=True)
    
    # Create environment
    env = CarEnv(render_mode=None)
    env = Monitor(env, LOGGING_CONFIG['log_dir'])
    
    # Create evaluation environment
    eval_env = CarEnv(render_mode=None)
    eval_env = Monitor(eval_env, LOGGING_CONFIG['log_dir'])
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=LOGGING_CONFIG['save_freq'],
        save_path=LOGGING_CONFIG['checkpoint_dir'],
        name_prefix='carv1_model'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=LOGGING_CONFIG['checkpoint_dir'],
        log_path=LOGGING_CONFIG['log_dir'],
        eval_freq=LOGGING_CONFIG['eval_freq'],
        n_eval_episodes=LOGGING_CONFIG['n_eval_episodes'],
        deterministic=True
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    # Add Hugging Face callback if enabled
    if HUGGINGFACE_CONFIG['enabled'] and HUGGINGFACE_CONFIG['repo_id']:
        print(f"\n🤗 Hugging Face integration enabled")
        print(f"   Repository: {HUGGINGFACE_CONFIG['repo_id']}")
        print(f"   Push frequency: every {HUGGINGFACE_CONFIG['push_freq']} steps")
        
        hf_callback = HuggingFaceCallback(
            repo_id=HUGGINGFACE_CONFIG['repo_id'],
            push_freq=HUGGINGFACE_CONFIG['push_freq'],
            checkpoint_dir=LOGGING_CONFIG['checkpoint_dir'],
            private=HUGGINGFACE_CONFIG['private'],
            verbose=1
        )
        callbacks.append(hf_callback)
    else:
        print("\n📝 Note: Hugging Face integration is disabled")
        print("   To enable, set HUGGINGFACE_CONFIG in config/config.py")
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=TRAINING_CONFIG['learning_rate'],
        n_steps=TRAINING_CONFIG['n_steps'],
        batch_size=TRAINING_CONFIG['batch_size'],
        n_epochs=TRAINING_CONFIG['n_epochs'],
        gamma=TRAINING_CONFIG['gamma'],
        gae_lambda=TRAINING_CONFIG['gae_lambda'],
        clip_range=TRAINING_CONFIG['clip_range'],
        ent_coef=TRAINING_CONFIG['ent_coef'],
        vf_coef=TRAINING_CONFIG['vf_coef'],
        max_grad_norm=TRAINING_CONFIG['max_grad_norm'],
        verbose=1,
        tensorboard_log=LOGGING_CONFIG['tensorboard_log']
    )
    
    print(f"\nTraining for {TRAINING_CONFIG['total_timesteps']:,} timesteps...")
    print(f"Tensorboard logs: {LOGGING_CONFIG['tensorboard_log']}")
    print(f"Checkpoints: {LOGGING_CONFIG['checkpoint_dir']}")
    print("\nTo monitor training, run:")
    print(f"  tensorboard --logdir {LOGGING_CONFIG['tensorboard_log']}")
    print()
    
    # Train
    model.learn(
        total_timesteps=TRAINING_CONFIG['total_timesteps'],
        callback=callbacks
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    
    # Save final model
    final_model_path = Path(LOGGING_CONFIG['checkpoint_dir']) / 'carv1_final_model'
    model.save(final_model_path)
    print(f"\n✓ Training complete! ({hours}h {minutes}m {seconds}s)")
    print(f"✓ Final model saved to: {final_model_path}")
    
    # Push to Hugging Face if enabled
    if HUGGINGFACE_CONFIG['enabled'] and HUGGINGFACE_CONFIG['repo_id'] and HUGGINGFACE_CONFIG['push_on_complete']:
        print(f"\n📤 Pushing final model to Hugging Face Hub...")
        
        push_final_model(
            model=model,
            repo_id=HUGGINGFACE_CONFIG['repo_id'],
            model_name="carv1_final_model",
            private=HUGGINGFACE_CONFIG['private'],
            commit_message=f"Final model after {TRAINING_CONFIG['total_timesteps']:,} timesteps"
        )
        
        # Create model card
        model_info = {
            'name': 'CarV1 Self-Driving Agent',
            'algorithm': 'PPO',
        }
        
        training_info = {
            'total_timesteps': TRAINING_CONFIG['total_timesteps'],
            'learning_rate': TRAINING_CONFIG['learning_rate'],
            'batch_size': TRAINING_CONFIG['batch_size'],
            'training_time': f"{hours}h {minutes}m {seconds}s",
            'hyperparameters': str(TRAINING_CONFIG),
        }
        
        create_model_card(
            repo_id=HUGGINGFACE_CONFIG['repo_id'],
            model_info=model_info,
            training_info=training_info
        )
        
        print(f"✓ View your model at: https://huggingface.co/{HUGGINGFACE_CONFIG['repo_id']}")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
