"""
Test trained model on challenging scenarios and record videos.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
from stable_baselines3 import PPO
from env.camera_line_follow_env import CameraLineFollowEnv
import cv2

def test_challenging_scenarios(model_path, output_dir="test_videos"):
    """Test model on various difficulty levels and record videos."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("CHALLENGING SCENARIO TESTING")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = PPO.load(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    # Define test scenarios
    scenarios = [
        {
            "name": "Easy - Wide Track (90cm)",
            "track_width_min": 0.9,
            "track_width_max": 0.9,
            "max_steps": 2000,
        },
        {
            "name": "Medium - Standard Track (60cm)",
            "track_width_min": 0.6,
            "track_width_max": 0.6,
            "max_steps": 2000,
        },
        {
            "name": "Hard - Narrow Track (40cm)",
            "track_width_min": 0.4,
            "track_width_max": 0.4,
            "max_steps": 2000,
        },
        {
            "name": "Very Hard - Tight Track (30cm)",
            "track_width_min": 0.3,
            "track_width_max": 0.3,
            "max_steps": 2000,
        },
        {
            "name": "Extreme - Variable Width (30-90cm)",
            "track_width_min": 0.3,
            "track_width_max": 0.9,
            "max_steps": 3000,
        },
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Testing: {scenario['name']}")
        print(f"{'='*60}")
        
        # Create environment for this scenario
        env = CameraLineFollowEnv(render_mode='rgb_array')
        env.unwrapped.track_width_min = scenario["track_width_min"]
        env.unwrapped.track_width_max = scenario["track_width_max"]
        env.unwrapped.max_steps = scenario["max_steps"]
        
        # Prepare video writer
        video_filename = output_path / f"{scenario['name'].replace(' ', '_').replace('-', '').lower()}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        
        # Run episode
        obs, info = env.reset()
        frame = env.render()
        height, width = frame.shape[:2]
        video = cv2.VideoWriter(str(video_filename), fourcc, fps, (width, height))
        
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Get action
            action, _ = model.predict(obs, deterministic=True)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # Record frame
            frame = env.render()
            # Add text overlay
            frame_with_text = frame.copy()
            cv2.putText(frame_with_text, scenario['name'], (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_with_text, f"Step: {steps}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_with_text, f"Reward: {episode_reward:.1f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            video.write(cv2.cvtColor(frame_with_text, cv2.COLOR_RGB2BGR))
        
        video.release()
        env.close()
        
        # Record results
        result = {
            "scenario": scenario['name'],
            "reward": episode_reward,
            "steps": steps,
            "completed": terminated,
            "video": str(video_filename)
        }
        results.append(result)
        
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {steps}/{scenario['max_steps']}")
        print(f"  Status: {'✓ Completed' if terminated else '⚠ Truncated/Failed'}")
        print(f"  Video: {video_filename}")
    
    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "✓" if r['completed'] else "✗"
        print(f"{status} {r['scenario']:40} | Reward: {r['reward']:7.1f} | Steps: {r['steps']:4}")
    
    print(f"\n✓ Videos saved to: {output_path.absolute()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model on challenging scenarios")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model/best_model.zip",
        help="Path to trained model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_videos",
        help="Output directory for videos"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    test_challenging_scenarios(str(model_path), args.output)
