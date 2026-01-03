"""
Visualize the camera-based line following environment with random actions.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from env.camera_line_follow_env import CameraLineFollowEnv


def main():
    """Run visualization with random agent."""
    env = CameraLineFollowEnv(render_mode="human")
    
    print("Starting visualization...")
    print("Close the window to exit.")
    print("\nRandom agent driving...")
    
    for episode in range(5):
        observation, info = env.reset()
        total_reward = 0
        
        print(f"\nEpisode {episode + 1}")
        
        for step in range(2000):
            # Random action
            action = env.action_space.sample()
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"  Steps: {step + 1}")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Final offset: {info['lateral_offset']:.2f} cm")
                print(f"  Still on track: {info['on_track']}")
                print(f"  Reason: {'Off track' if terminated else 'Max steps'}")
                break
    
    env.close()
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
