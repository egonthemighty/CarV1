"""
Capture a still image from the first-person environment for analysis
"""
import sys
sys.path.append('.')

from env.first_person_env import FirstPersonLineFollowEnv
import cv2
import numpy as np

def capture_views():
    """Capture multiple views at different track positions"""
    env = FirstPersonLineFollowEnv(render_mode="rgb_array")
    
    print("Capturing views from first-person environment...")
    print(f"Camera specs:")
    print(f"  Position: {env.camera_x_offset*100:.1f}cm from front hub")
    print(f"  Height: {env.camera_height*100:.1f}cm above ground")
    print(f"  Pitch: {env.camera_pitch:.1f}°")
    print(f"  FOV: {env.camera_fov:.1f}°")
    
    # Capture at different positions
    scenarios = [
        ("centered", 0.0, 0.0),
        ("slightly_left", 0.1, 0.0),
        ("slightly_right", -0.1, 0.0),
    ]
    
    for name, x_offset, y_offset in scenarios:
        # Reset and set specific position
        obs, info = env.reset()
        env.car_x = x_offset
        env.car_y = y_offset
        env.car_theta = 0.0  # Facing forward
        
        # Render the view
        frame = env.render()
        
        # Save as image
        filename = f"view_{name}.png"
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, frame_bgr)
        print(f"✓ Saved {filename}")
        
        # Also create an annotated version
        annotated = frame_bgr.copy()
        
        # Add text overlay with camera info
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 30
        texts = [
            f"Camera: {env.camera_height*100:.1f}cm high, {abs(env.camera_x_offset)*100:.1f}cm from front hub",
            f"Pitch: {env.camera_pitch:.1f}deg, FOV: {env.camera_fov:.1f}deg",
            f"Car position: X={x_offset:.2f}m, Y={y_offset:.2f}m",
            f"Track width: {env.track_width_min:.2f}-{env.track_width_max:.2f}m (variable)",
        ]
        
        for text in texts:
            cv2.putText(annotated, text, (10, y_pos), font, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated, text, (10, y_pos), font, 0.6, (0, 0, 0), 1)
            y_pos += 30
        
        annotated_filename = f"view_{name}_annotated.png"
        cv2.imwrite(annotated_filename, annotated)
        print(f"✓ Saved {annotated_filename}")
    
    env.close()
    print("\nAll views captured! Check the PNG files in the current directory.")

if __name__ == "__main__":
    capture_views()
