"""
Test the first-person camera environment to verify it works before training.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from env.first_person_env import FirstPersonLineFollowEnv

def test_environment():
    """Test the first-person environment with random actions."""
    print("=" * 60)
    print("TESTING FIRST-PERSON CAMERA ENVIRONMENT")
    print("=" * 60)
    
    # Create environment
    env = FirstPersonLineFollowEnv(render_mode='human')
    
    print("\nEnvironment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("\nRunning test episode with random actions...")
    print("Watch the pygame window - you should see first-person view!")
    print("Press Ctrl+C to stop\n")
    
    try:
        obs, info = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        while not done and steps < 500:
            # Random action
            action = env.action_space.sample()
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            # Render
            env.render()
            time.sleep(0.03)  # ~30 FPS
            
            # Print status every 50 steps
            if steps % 50 == 0:
                print(f"Step {steps}: Reward={total_reward:.1f}, "
                      f"Offset={info['lateral_offset']:.2f}m, "
                      f"Speed={info['speed']:.1f}m/s")
        
        print(f"\nEpisode finished!")
        print(f"Total steps: {steps}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Reason: {'Off track' if terminated else 'Time limit'}")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    finally:
        env.close()
        print("\n✓ Environment test complete!")

if __name__ == "__main__":
    test_environment()
