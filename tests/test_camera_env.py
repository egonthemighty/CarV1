"""
Test script for camera-based line following environment.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from env.camera_line_follow_env import CameraLineFollowEnv


def test_env_creation():
    """Test environment can be created."""
    print("Testing environment creation...")
    env = CameraLineFollowEnv()
    print("✓ Environment created successfully")
    env.close()


def test_env_reset():
    """Test environment reset."""
    print("\nTesting environment reset...")
    env = CameraLineFollowEnv()
    observation, info = env.reset()
    
    assert observation is not None, "Observation is None"
    print(f"✓ Environment reset successfully")
    print(f"  Observation shape: {observation.shape}")
    print(f"  Observation: {observation}")
    print(f"  Info: {info}")
    env.close()


def test_env_step():
    """Test environment step."""
    print("\nTesting environment step...")
    env = CameraLineFollowEnv()
    observation, info = env.reset()
    
    # Take action: straight forward
    action = [0.0, 0.5]  # No steering, half throttle
    observation, reward, terminated, truncated, info = env.step(action)
    
    assert observation is not None, "Observation is None"
    assert isinstance(reward, float), "Reward is not a float"
    assert isinstance(terminated, bool), "Terminated is not a bool"
    assert isinstance(truncated, bool), "Truncated is not a bool"
    
    print(f"✓ Environment step successful")
    print(f"  Action: {action}")
    print(f"  Reward: {reward:.3f}")
    print(f"  Terminated: {terminated}, Truncated: {truncated}")
    print(f"  Info: {info}")
    env.close()


def test_straight_driving():
    """Test driving straight."""
    print("\nTesting straight driving...")
    env = CameraLineFollowEnv()
    observation, info = env.reset()
    
    total_reward = 0
    steps = 0
    max_steps = 100
    
    print("  Driving straight with no steering...")
    for step in range(max_steps):
        # Drive straight
        action = [0.0, 0.3]  # No steering, gentle throttle
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    print(f"✓ Straight driving completed")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final offset: {info['lateral_offset']:.2f} cm")
    print(f"  Still on track: {info['on_track']}")
    env.close()


def test_observation_modes():
    """Test both observation modes."""
    print("\nTesting observation modes...")
    
    # Feature-based
    env_features = CameraLineFollowEnv(use_raw_pixels=False)
    obs_features, _ = env_features.reset()
    print(f"✓ Feature-based observations")
    print(f"  Shape: {obs_features.shape}")
    print(f"  Type: {obs_features.dtype}")
    print(f"  Values: {obs_features}")
    env_features.close()
    
    # Raw pixels
    env_pixels = CameraLineFollowEnv(use_raw_pixels=True, camera_resolution=(84, 84))
    obs_pixels, _ = env_pixels.reset()
    print(f"✓ Raw pixel observations")
    print(f"  Shape: {obs_pixels.shape}")
    print(f"  Type: {obs_pixels.dtype}")
    print(f"  Range: [{obs_pixels.min()}, {obs_pixels.max()}]")
    env_pixels.close()


def test_spaces():
    """Test observation and action spaces."""
    print("\nTesting spaces...")
    env = CameraLineFollowEnv()
    
    print(f"✓ Observation space: {env.observation_space}")
    print(f"  Shape: {env.observation_space.shape}")
    
    print(f"✓ Action space: {env.action_space}")
    print(f"  Shape: {env.action_space.shape}")
    print(f"  Low: {env.action_space.low}")
    print(f"  High: {env.action_space.high}")
    
    env.close()


if __name__ == "__main__":
    print("="*60)
    print("Camera Line Follow Environment Test Suite")
    print("="*60)
    
    try:
        test_env_creation()
        test_env_reset()
        test_env_step()
        test_straight_driving()
        test_observation_modes()
        test_spaces()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run visualization: python tests/visualize_camera_env.py")
        print("  2. Start training: python tests/train_camera.py")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
