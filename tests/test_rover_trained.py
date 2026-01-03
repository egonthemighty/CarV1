"""
Test Rover's trained model with first-person camera view
"""
import sys
sys.path.append('.')

from env.first_person_env import FirstPersonLineFollowEnv
from stable_baselines3 import PPO
import numpy as np

def test_trained_model(model_path, num_episodes=3):
    """Test the trained model and visualize Rover's performance."""
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment with human rendering
    env = FirstPersonLineFollowEnv(render_mode="human")
    
    print(f"\nTesting Rover for {num_episodes} episodes...")
    print("Watch the pygame window - you'll see:")
    print("  - Clean road view (what Rover's camera sees)")
    print("  - Depth bands (human visual aid)")
    print("  - Hood and collision indicators (overlays)")
    print("\nPress Ctrl+C to stop\n")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1}")
        print(f"{'='*60}")
        
        while not done:
            # Use trained model to predict action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            # Print progress every 200 steps
            if steps % 200 == 0:
                print(f"Step {steps}: Reward={total_reward:.1f}, "
                      f"Offset={info['lateral_offset']:.3f}m, "
                      f"Speed={info['speed']:.2f}m/s, "
                      f"Heading={np.degrees(info['heading']):.1f}°")
        
        # Episode summary
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1} COMPLETE")
        print(f"{'='*60}")
        print(f"Total Steps:  {steps}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Avg Reward:   {total_reward/steps:.4f}")
        print(f"Final Speed:  {info['speed']:.2f} m/s")
        print(f"Final Offset: {info['lateral_offset']:.3f} m")
        print(f"Track Width:  {info['track_width']:.2f} m")
        
        if terminated:
            left_margin = info.get('left_margin', 0)
            right_margin = info.get('right_margin', 0)
            print(f"\nTermination: Off track")
            print(f"  Left margin:  {left_margin:.3f} m")
            print(f"  Right margin: {right_margin:.3f} m")
        else:
            print(f"\nCompleted: Max steps reached")
    
    env.close()
    print("\n✓ Testing complete!")

if __name__ == "__main__":
    # Test the final model
    model_path = "training output/carv1_models/ppo_camera_line_follow_final.zip"
    test_trained_model(model_path, num_episodes=3)
