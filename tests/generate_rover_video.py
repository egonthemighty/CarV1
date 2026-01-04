"""
Generate video visualization of Rover's trained model performance
"""
import sys
sys.path.append('.')

from env.first_person_env import FirstPersonLineFollowEnv
from stable_baselines3 import PPO
import numpy as np
import cv2
import os

def generate_video(model_path, output_folder, num_episodes=1):
    """Generate video of Rover's performance."""
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment with rgb_array rendering and raw pixels enabled
    env = FirstPersonLineFollowEnv(
        render_mode="rgb_array",
        use_raw_pixels=True,
        camera_resolution=(84, 84)
    )
    
    for episode in range(num_episodes):
        print(f"\nGenerating video for episode {episode + 1}...")
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Setup video writer
        output_path = os.path.join(output_folder, f"rover_episode_{episode+1}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        frame = env.render()
        height, width = frame.shape[:2]
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Recording to: {output_path}")
        print(f"Resolution: {width}x{height} @ {fps}fps")
        
        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            # Render and write frame
            frame = env.render()
            
            # Add telemetry overlay
            overlay_frame = frame.copy()
            
            # Add text info
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_pos = 30
            texts = [
                f"Rover Performance - Episode {episode+1}",
                f"Step: {steps}/2000",
                f"Reward: {total_reward:.1f}",
                f"Offset: {info['lateral_offset']:.3f}m",
                f"Speed: {info['speed']:.2f}m/s",
                f"Heading: {np.degrees(info['heading']):.1f}deg",
                f"Track Width: {info['track_width']:.2f}m",
            ]
            
            for text in texts:
                # Shadow
                cv2.putText(overlay_frame, text, (12, y_pos+2), font, 0.6, (0, 0, 0), 2)
                # Text
                cv2.putText(overlay_frame, text, (10, y_pos), font, 0.6, (255, 255, 255), 1)
                y_pos += 30
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            
            # Print progress
            if steps % 200 == 0:
                print(f"  Step {steps}: Reward={total_reward:.1f}, Offset={info['lateral_offset']:.3f}m")
        
        video_writer.release()
        
        # Episode summary
        print(f"\n{'='*60}")
        print(f"Video saved: {output_path}")
        print(f"Total Steps: {steps}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Avg Reward: {total_reward/steps:.4f}")
        print(f"Final Offset: {info['lateral_offset']:.3f}m")
        if terminated:
            print(f"Result: Went off track")
        else:
            print(f"Result: Completed successfully")
        print(f"{'='*60}")
    
    env.close()
    print(f"\n✓ Video generation complete!")
    print(f"✓ Saved to: {output_folder}")

if __name__ == "__main__":
    model_path = "training output/ppo_carv1_final.zip"
    output_folder = "training output/videos"
    
    generate_video(model_path, output_folder, num_episodes=1)
