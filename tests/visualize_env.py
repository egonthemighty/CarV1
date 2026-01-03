"""
Simple script to visualize the CarV1 environment with random actions.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from env.car_env import CarEnv


def main():
    """Run visualization with random agent."""
    env = CarEnv(render_mode="human")
    
    print("Starting visualization...")
    print("Close the window to exit.")
    
    for episode in range(5):
        observation, info = env.reset()
        total_reward = 0
        
        print(f"\nEpisode {episode + 1}")
        
        for step in range(1000):
            # Random action
            action = env.action_space.sample()
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"  Steps: {step + 1}")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Final position: {info['position']}")
                print(f"  Reason: {'Collision' if terminated else 'Max steps'}")
                break
    
    env.close()
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
