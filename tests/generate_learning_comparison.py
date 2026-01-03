"""
Generate 2x2 comparison video showing Rover's learning progress
Shows 4 training checkpoints side-by-side: 250k, 500k, 750k, and 1M timesteps
"""
import sys
sys.path.append('.')

from env.first_person_env import FirstPersonLineFollowEnv
from stable_baselines3 import PPO
import numpy as np
import cv2
import os

def generate_comparison_video(checkpoint_steps, output_folder, max_steps=500):
    """Generate 2x2 comparison video of different training checkpoints."""
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load all 4 models
    models = []
    labels = []
    for steps in checkpoint_steps:
        if steps == 1000000:
            model_path = "training output/carv1_models/ppo_camera_line_follow_final.zip"
            label = f"Final (1M)"
        else:
            model_path = f"training output/carv1_models/checkpoints/ppo_camera_line_follow_{steps}_steps.zip"
            label = f"{steps//1000}k steps"
        
        print(f"Loading model: {label}")
        models.append(PPO.load(model_path))
        labels.append(label)
    
    # Create 4 environments
    print(f"\nCreating 4 environments...")
    envs = [FirstPersonLineFollowEnv(render_mode="rgb_array") for _ in range(4)]
    
    # Reset all environments with same seed for fair comparison
    seed = 42
    observations = [env.reset(seed=seed)[0] for env in envs]
    
    # Setup video writer
    output_path = os.path.join(output_folder, "rover_learning_progress_2x2.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    
    # Get single frame size
    frame = envs[0].render()
    single_height, single_width = frame.shape[:2]
    
    # 2x2 grid size
    grid_width = single_width * 2
    grid_height = single_height * 2
    
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))
    
    print(f"Recording to: {output_path}")
    print(f"Resolution: {grid_width}x{grid_height} @ {fps}fps")
    print(f"Running for {max_steps} steps...")
    
    # Track state for all 4 agents
    done_flags = [False] * 4
    rewards = [0.0] * 4
    steps = 0
    
    while steps < max_steps and not all(done_flags):
        # Create 2x2 grid frame
        grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        for i, (model, env, obs, done) in enumerate(zip(models, envs, observations, done_flags)):
            if not done:
                # Predict action
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                observations[i] = obs
                rewards[i] += reward
                
                if terminated or truncated:
                    done_flags[i] = True
            
            # Render frame
            frame = env.render()
            
            # Add label and stats overlay
            overlay_frame = frame.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Title
            cv2.putText(overlay_frame, labels[i], (12, 32), font, 0.8, (0, 0, 0), 3)
            cv2.putText(overlay_frame, labels[i], (10, 30), font, 0.8, (255, 255, 255), 2)
            
            # Stats
            if not done_flags[i]:
                stats = [
                    f"Step: {steps}",
                    f"Reward: {rewards[i]:.1f}",
                    f"Offset: {info['lateral_offset']:.3f}m",
                ]
            else:
                stats = [
                    f"FINISHED",
                    f"Final: {rewards[i]:.1f}",
                ]
            
            y_pos = single_height - 70
            for stat in stats:
                cv2.putText(overlay_frame, stat, (12, y_pos+2), font, 0.5, (0, 0, 0), 2)
                cv2.putText(overlay_frame, stat, (10, y_pos), font, 0.5, (255, 255, 255), 1)
                y_pos += 25
            
            # Place in grid (top-left, top-right, bottom-left, bottom-right)
            row = i // 2
            col = i % 2
            y_start = row * single_height
            x_start = col * single_width
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR)
            grid_frame[y_start:y_start+single_height, x_start:x_start+single_width] = frame_bgr
        
        # Write frame
        video_writer.write(grid_frame)
        
        steps += 1
        
        # Print progress
        if steps % 50 == 0:
            print(f"  Step {steps}/{max_steps}")
            for i, label in enumerate(labels):
                status = "DONE" if done_flags[i] else f"Running (R={rewards[i]:.0f})"
                print(f"    {label}: {status}")
    
    video_writer.release()
    
    # Close environments
    for env in envs:
        env.close()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"COMPARISON VIDEO COMPLETE")
    print(f"{'='*60}")
    print(f"Video saved: {output_path}")
    print(f"Total frames: {steps}")
    print(f"\nFinal Performance:")
    for i, label in enumerate(labels):
        status = "Completed" if steps >= max_steps else "Off track"
        print(f"  {label}: {rewards[i]:.1f} reward ({status})")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Compare 4 checkpoints: 250k, 500k, 750k, 1M
    checkpoint_steps = [250000, 500000, 750000, 1000000]
    output_folder = "training output/videos"
    
    # Run for 500 steps (about 16 seconds at 30fps)
    generate_comparison_video(checkpoint_steps, output_folder, max_steps=500)
