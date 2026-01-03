"""
Test script to verify the CarV1 environment works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gymnasium as gym
from env.car_env import CarEnv


def test_env_creation():
    """Test environment can be created."""
    print("Testing environment creation...")
    env = CarEnv()
    print("✓ Environment created successfully")
    env.close()


def test_env_reset():
    """Test environment reset."""
    print("\nTesting environment reset...")
    env = CarEnv()
    observation, info = env.reset()
    
    assert observation is not None, "Observation is None"
    assert len(observation) == 6 + 8, f"Expected 14 observations, got {len(observation)}"
    print(f"✓ Environment reset successfully")
    print(f"  Observation shape: {observation.shape}")
    print(f"  Info: {info}")
    env.close()


def test_env_step():
    """Test environment step."""
    print("\nTesting environment step...")
    env = CarEnv()
    observation, info = env.reset()
    
    # Take random action
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    assert observation is not None, "Observation is None"
    assert isinstance(reward, float), "Reward is not a float"
    assert isinstance(terminated, bool), "Terminated is not a bool"
    assert isinstance(truncated, bool), "Truncated is not a bool"
    
    print(f"✓ Environment step successful")
    print(f"  Action: {action}")
    print(f"  Reward: {reward}")
    print(f"  Terminated: {terminated}, Truncated: {truncated}")
    env.close()


def test_full_episode():
    """Test a full episode with random actions."""
    print("\nTesting full episode...")
    env = CarEnv()
    observation, info = env.reset()
    
    total_reward = 0
    steps = 0
    max_steps = 100
    
    for step in range(max_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    print(f"✓ Episode completed")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final position: ({info['position'][0]:.1f}, {info['position'][1]:.1f})")
    print(f"  Final speed: {info['speed']:.2f}")
    env.close()


def test_spaces():
    """Test observation and action spaces."""
    print("\nTesting spaces...")
    env = CarEnv()
    
    print(f"✓ Observation space: {env.observation_space}")
    print(f"  Shape: {env.observation_space.shape}")
    print(f"  Low: {env.observation_space.low}")
    print(f"  High: {env.observation_space.high}")
    
    print(f"✓ Action space: {env.action_space}")
    print(f"  Shape: {env.action_space.shape}")
    print(f"  Low: {env.action_space.low}")
    print(f"  High: {env.action_space.high}")
    
    env.close()


if __name__ == "__main__":
    print("="*60)
    print("CarV1 Environment Test Suite")
    print("="*60)
    
    try:
        test_env_creation()
        test_env_reset()
        test_env_step()
        test_full_episode()
        test_spaces()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
