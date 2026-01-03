"""
Test and visualize the trained camera line-following model.
Loads a trained model and runs it in the environment with rendering.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from stable_baselines3 import PPO
from env.camera_line_follow_env import CameraLineFollowEnv

def test_model(model_path, num_episodes=5, render=True):
    """
    Test a trained model and visualize its performance.
    
    Args:
        model_path: Path to the trained model (.zip file)
        num_episodes: Number of episodes to run
        render: Whether to render the environment
    """
    print("=" * 60)
    print("TESTING TRAINED MODEL")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)
    
    # Load the trained model
    print("\nLoading model...")
    model = PPO.load(model_path)
    print(f"✓ Model loaded successfully")
    print(f"  Device: {model.device}")
    
    # Create environment with rendering
    render_mode = 'human' if render else None
    env = CameraLineFollowEnv(render_mode=render_mode)
    
    # Test for multiple episodes
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # Render if enabled
            if render:
                env.render()
                time.sleep(0.01)  # Slow down a bit for viewing
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Reason: {'Completed' if terminated else 'Truncated'}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Average Reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"Average Steps: {sum(episode_lengths)/len(episode_lengths):.1f}")
    print(f"Best Reward: {max(episode_rewards):.2f}")
    print(f"Worst Reward: {min(episode_rewards):.2f}")
    print("=" * 60)
    
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trained CarV1 model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model/best_model.zip",
        help="Path to trained model (default: models/best_model/best_model.zip)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to test (default: 5)"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering (run headless)"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("\nAvailable models:")
        
        # Look for models
        models_dir = Path("models")
        if models_dir.exists():
            for model_file in models_dir.rglob("*.zip"):
                print(f"  - {model_file}")
        else:
            print("  No models directory found")
        
        print("\nExtract your trained models from carv1_models.zip first!")
        sys.exit(1)
    
    # Test the model
    test_model(
        model_path=str(model_path),
        num_episodes=args.episodes,
        render=not args.no_render
    )
