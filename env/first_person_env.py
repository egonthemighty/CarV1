"""
First-Person Camera Line Following Environment

Matches real hardware setup:
- Camera mounted on car (1 foot back from front, 10 inches high)
- Forward-looking perspective view
- Sees track ahead, not top-down
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import cv2
import math

# HARDWARE CONFIGURATION - Adjust these to match physical build
CAR_LENGTH = 0.61  # 24 inches = 0.61m (total length with bumpers)
WHEELBASE_LENGTH = 0.33  # 13 inches = 0.33m (hub to hub)
WHEELBASE_WIDTH = 0.244  # 9.625 inches = 0.244m (center to center)
CAMERA_HEIGHT = 0.254  # 10 inches = 0.254m above ground
CAMERA_X_OFFSET = -0.165  # 6.5 inches from front hub (centered on wheelbase)
CAMERA_PITCH = -11.8  # degrees (focused 48" ahead: arctan(10/48))
CAMERA_FOV = 60.0  # degrees horizontal field of view


class FirstPersonLineFollowEnv(gym.Env):
    """
    Environment with first-person camera perspective matching Pi Camera setup.
    
    Actor: "Rover" - the autonomous RC car
    
    Hardware Specs (configurable at top of file):
    - Car: 24" long (with bumpers), 13" wheelbase, 9.625" wheel track
    - Camera: 6.5" from front hub (centered on wheelbase)
    - Camera: 10" above ground, pitched 11.8° down (focused 48" ahead)
    
    Observation: Forward-looking camera view showing track ahead
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        render_mode=None,
        use_raw_pixels=False,
        camera_resolution=(84, 84),
        track_width_min=0.6,  # meters
        track_width_max=0.6,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.use_raw_pixels = use_raw_pixels
        self.camera_resolution = camera_resolution
        
        # Track parameters (meters)
        self.track_width_min = track_width_min
        self.track_width_max = track_width_max
        self.current_track_width = track_width_min
        
        # Car parameters (from hardware configuration)
        self.car_length = CAR_LENGTH
        self.car_width = WHEELBASE_WIDTH  # Use wheel track width for collision detection
        self.wheelbase = WHEELBASE_LENGTH
        
        # Camera parameters (from hardware configuration)
        self.camera_height = CAMERA_HEIGHT
        self.camera_x_offset = CAMERA_X_OFFSET
        self.camera_pitch = CAMERA_PITCH
        self.camera_fov = CAMERA_FOV
        
        # Physics parameters
        self.max_speed = 2.0  # m/s
        self.max_steering_angle = np.deg2rad(45)
        self.dt = 0.033  # ~30 FPS
        
        # View parameters for rendering first-person view
        self.view_distance = 3.0  # meters ahead to render
        self.view_width = 2.0  # meters of track width in view
        
        # Display parameters
        self.display_width = 800
        self.display_height = 600
        
        # Episode parameters
        self.max_steps = 2000
        
        # Define observation space
        if use_raw_pixels:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(camera_resolution[0], camera_resolution[1], 1),
                dtype=np.uint8
            )
        else:
            # Features: [left_line_offset, right_line_offset, heading, speed]
            # Offsets in meters, heading in radians
            self.observation_space = spaces.Box(
                low=np.array([-1.5, -1.5, -np.pi, 0], dtype=np.float32),
                high=np.array([1.5, 1.5, np.pi, self.max_speed], dtype=np.float32),
                dtype=np.float32
            )
        
        # Action space: [steering, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Pygame/rendering
        self.window = None
        self.clock = None
        self.font = None
        
        # State
        self.state = None
        self.steps = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize track width
        if self.track_width_min != self.track_width_max:
            self.current_track_width = self.np_random.uniform(
                self.track_width_min, self.track_width_max
            )
        else:
            self.current_track_width = self.track_width_min
        
        # Initialize car state
        # Position: lateral offset from track center (meters)
        # Heading: angle relative to track direction (radians)
        # Speed: forward velocity (m/s)
        self.state = {
            'lateral_offset': 0.0,  # centered between lines
            'heading': 0.0,  # aligned with track
            'speed': 0.5,  # starting speed
            'distance': 0.0,  # distance traveled
        }
        
        self.steps = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        self.steps += 1
        
        # Extract actions
        steering = np.clip(action[0], -1, 1) * self.max_steering_angle
        throttle = np.clip(action[1], -1, 1)
        
        # Update speed
        acceleration = throttle * 3.0  # m/s²
        self.state['speed'] = np.clip(
            self.state['speed'] + acceleration * self.dt,
            0.0,
            self.max_speed
        )
        
        # Update heading (bicycle model)
        if abs(self.state['speed']) > 0.01:
            turn_rate = self.state['speed'] * np.tan(steering) / self.car_length
            self.state['heading'] += turn_rate * self.dt
            self.state['heading'] = np.arctan2(
                np.sin(self.state['heading']),
                np.cos(self.state['heading'])
            )
        
        # Update lateral position
        lateral_velocity = self.state['speed'] * np.sin(self.state['heading'])
        self.state['lateral_offset'] += lateral_velocity * self.dt
        
        # Update forward distance
        forward_velocity = self.state['speed'] * np.cos(self.state['heading'])
        self.state['distance'] += forward_velocity * self.dt
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination (Rover going off track)
        # Collision uses wheelbase width (wheel center to wheel center)
        half_width = self.current_track_width / 2
        left_line = -half_width
        right_line = half_width
        
        # Rover is off-track when wheels cross the boundary lines
        off_track = (
            self.state['lateral_offset'] < left_line - self.car_width/2 or
            self.state['lateral_offset'] > right_line + self.car_width/2
        )
        
        terminated = off_track
        truncated = self.steps >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get observation from car's first-person camera view."""
        if self.use_raw_pixels:
            # Render first-person camera view
            image = self._render_camera_view()
            # Convert to grayscale and resize
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, self.camera_resolution)
            return resized.reshape((*self.camera_resolution, 1))
        else:
            # Extract features from camera view
            return self._extract_features()
    
    def _extract_features(self):
        """Extract feature representation from camera view."""
        # Calculate line positions relative to car center
        half_width = self.current_track_width / 2
        left_line_offset = -half_width - self.state['lateral_offset']
        right_line_offset = half_width - self.state['lateral_offset']
        
        return np.array([
            left_line_offset,
            right_line_offset,
            self.state['heading'],
            self.state['speed']
        ], dtype=np.float32)
    
    def _calculate_reward(self):
        """Calculate reward based on staying centered and moving forward."""
        # Centering reward (higher when closer to center)
        center_distance = abs(self.state['lateral_offset'])
        max_offset = self.current_track_width / 2
        centering_reward = 1.0 - (center_distance / max_offset)
        
        # Forward progress reward
        progress_reward = self.state['speed'] * np.cos(self.state['heading']) * 0.5
        
        # Heading alignment reward
        heading_reward = (1.0 - abs(self.state['heading']) / np.pi) * 0.3
        
        # Penalty for going off track
        if center_distance > max_offset:
            return -10.0
        
        return centering_reward + progress_reward + heading_reward
    
    def _get_info(self):
        """Get additional info dict."""
        half_width = self.current_track_width / 2
        half_car = self.car_width / 2  # Uses wheelbase width (9.625")
        
        # Calculate margins (distance from wheel center to track edge)
        left_margin = (half_width + self.state['lateral_offset']) - half_car
        right_margin = (half_width - self.state['lateral_offset']) - half_car
        
        return {
            'lateral_offset': self.state['lateral_offset'],
            'heading': self.state['heading'],
            'speed': self.state['speed'],
            'distance': self.state['distance'],
            'track_width': self.current_track_width,
            'left_line_distance': half_width + self.state['lateral_offset'],
            'right_line_distance': half_width - self.state['lateral_offset'],
            'left_margin': left_margin,  # Space before left wheel crosses boundary
            'right_margin': right_margin,  # Space before right wheel crosses boundary
            'collision_detection': 'wheelbase_width (9.625" wheel track)',
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        return self._render_frame()
    
    def _render_frame(self):
        """Render a frame showing first-person view with overlays for human viewing."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.display_width, self.display_height))
            pygame.display.set_caption("CarV1 - Rover First-Person View")
            self.font = pygame.font.Font(None, 36)
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Create surface
        canvas = pygame.Surface((self.display_width, self.display_height))
        
        # Draw first-person perspective view (clean road and track)
        self._draw_first_person_view(canvas)
        
        # Draw visual aids for human viewing (NOT in Rover's observation)
        self._draw_depth_bands(canvas)  # Alternating road segments for depth perception
        
        # Draw overlays for human viewing
        self._draw_rover_overlays(canvas)
        
        # Draw HUD overlay
        self._draw_hud(canvas)
        
        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def _draw_first_person_view(self, canvas):
        """Draw clean road and track view (what Rover's camera actually sees)."""
        width = canvas.get_width()
        height = canvas.get_height()
        
        # Camera parameters for perspective projection
        cam_height = self.camera_height
        cam_pitch_rad = np.deg2rad(self.camera_pitch)
        fov_rad = np.deg2rad(self.camera_fov)
        
        # Calculate horizon position based on camera pitch
        horizon_y = int(height * 0.3)  # Horizon near top third
        
        # Draw uniform sky/ground (no depth bands - that's for human viewing)
        pygame.draw.rect(canvas, (135, 206, 235), (0, 0, width, horizon_y))  # Sky blue
        pygame.draw.rect(canvas, (55, 55, 55), (0, horizon_y, width, height))  # Uniform ground
        
        # Draw track lines (white ropes) with proper perspective projection
        half_track = self.current_track_width / 2
        
        # Project points from 3D world to 2D screen
        segments = 30
        for i in range(segments):
            # Distance ahead in world coordinates
            dist_near = (i / segments) * self.view_distance
            dist_far = ((i + 1) / segments) * self.view_distance
            
            # Skip the very closest segment (would be behind camera)
            if dist_near < 0.1:
                continue
            
            # Project near and far points
            def project_point(distance, lateral_offset):
                """Project a 3D point to 2D screen coordinates."""
                # 3D position relative to camera
                z = distance  # Distance ahead
                x = lateral_offset - self.state['lateral_offset']  # Lateral offset
                y = -cam_height  # Ground is below camera
                
                # Apply camera pitch rotation
                z_rot = z * np.cos(-cam_pitch_rad) - y * np.sin(-cam_pitch_rad)
                y_rot = z * np.sin(-cam_pitch_rad) + y * np.cos(-cam_pitch_rad)
                
                # Perspective projection
                if z_rot <= 0.01:  # Too close or behind camera
                    return None
                
                # Screen coordinates
                screen_x = width / 2 + (x / z_rot) * (width / (2 * np.tan(fov_rad / 2)))
                screen_y = height / 2 - (y_rot / z_rot) * (width / (2 * np.tan(fov_rad / 2)))
                
                return (screen_x, screen_y)
            
            # Project track edges at near and far distances
            left_near = project_point(dist_near, -half_track)
            right_near = project_point(dist_near, half_track)
            left_far = project_point(dist_far, -half_track)
            right_far = project_point(dist_far, half_track)
            
            if not all([left_near, right_near, left_far, right_far]):
                continue
            
            # Draw white line markers (ropes) - this is what Rover actually sees
            line_thickness = max(2, int(10 / (1 + dist_near * 0.5)))
            pygame.draw.line(canvas, (255, 255, 255), left_near, left_far, line_thickness)
            pygame.draw.line(canvas, (255, 255, 255), right_near, right_far, line_thickness)
    
    def _draw_depth_bands(self, canvas):
        """Draw alternating road bands for human depth perception (NOT in Rover's view)."""
        width = canvas.get_width()
        height = canvas.get_height()
        
        cam_height = self.camera_height
        cam_pitch_rad = np.deg2rad(self.camera_pitch)
        fov_rad = np.deg2rad(self.camera_fov)
        half_track = self.current_track_width / 2
        
        segments = 30
        for i in range(segments):
            dist_near = (i / segments) * self.view_distance
            dist_far = ((i + 1) / segments) * self.view_distance
            
            if dist_near < 0.1:
                continue
            
            def project_point(distance, lateral_offset):
                z = distance
                x = lateral_offset - self.state['lateral_offset']
                y = -cam_height
                
                z_rot = z * np.cos(-cam_pitch_rad) - y * np.sin(-cam_pitch_rad)
                y_rot = z * np.sin(-cam_pitch_rad) + y * np.cos(-cam_pitch_rad)
                
                if z_rot <= 0.01:
                    return None
                
                screen_x = width / 2 + (x / z_rot) * (width / (2 * np.tan(fov_rad / 2)))
                screen_y = height / 2 - (y_rot / z_rot) * (width / (2 * np.tan(fov_rad / 2)))
                return (screen_x, screen_y)
            
            left_near = project_point(dist_near, -half_track)
            right_near = project_point(dist_near, half_track)
            left_far = project_point(dist_far, -half_track)
            right_far = project_point(dist_far, half_track)
            
            if not all([left_near, right_near, left_far, right_far]):
                continue
            
            # Draw alternating dark bands (for human depth perception only)
            if i % 2 == 0:
                band_color = (45, 45, 45, 100)  # Slightly darker with transparency
                # Create surface with alpha for blending
                band_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                pygame.draw.polygon(band_surface, band_color, [
                    left_near, right_near, right_far, left_far
                ])
                canvas.blit(band_surface, (0, 0))
    
    def _draw_rover_overlays(self, canvas):
        """Draw Rover's hood and collision indicators (for human viewing only)."""
        self._draw_vehicle_hood(canvas)
    
    def _draw_vehicle_hood(self, canvas):
        """Draw Rover's hood/front in bottom of view."""
        width = self.display_width
        height = self.display_height
        
        # Hood takes up bottom portion of screen
        hood_height = int(height * 0.25)  # Bottom 25% of screen
        hood_top = height - hood_height
        
        # Draw hood as trapezoid (wider at bottom, narrower at top due to perspective)
        hood_width_bottom = width
        hood_width_top = int(width * 0.7)  # Narrower at top
        
        left_bottom = 0
        right_bottom = width
        left_top = (width - hood_width_top) // 2
        right_top = left_top + hood_width_top
        
        # Draw hood body
        hood_color = (40, 40, 45)  # Dark gray/black
        pygame.draw.polygon(canvas, hood_color, [
            (left_bottom, height),
            (right_bottom, height),
            (right_top, hood_top),
            (left_top, hood_top),
        ])
        
        # Draw hood edge highlight
        pygame.draw.line(canvas, (60, 60, 70), 
                        (left_top, hood_top), (right_top, hood_top), 3)
        
        # Visualize Rover's collision boundaries
        # Show left and right edges of car relative to track
        half_car_width = self.car_width / 2
        half_track = self.current_track_width / 2
        
        # Project car edges onto screen
        cam_height = self.camera_height
        cam_pitch_rad = np.deg2rad(self.camera_pitch)
        fov_rad = np.deg2rad(self.camera_fov)
        
        # Distance where we visualize boundaries (2m ahead)
        viz_distance = 2.0
        
        def project_boundary(lateral_offset):
            z = viz_distance
            x = lateral_offset - self.state['lateral_offset']
            y = -cam_height
            
            z_rot = z * np.cos(-cam_pitch_rad) - y * np.sin(-cam_pitch_rad)
            y_rot = z * np.sin(-cam_pitch_rad) + y * np.cos(-cam_pitch_rad)
            
            if z_rot <= 0.01:
                return None
            
            screen_x = width / 2 + (x / z_rot) * (width / (2 * np.tan(fov_rad / 2)))
            screen_y = height / 2 - (y_rot / z_rot) * (width / (2 * np.tan(fov_rad / 2)))
            return (screen_x, screen_y)
        
        # Car edge positions
        left_car_edge = project_boundary(-half_car_width)
        right_car_edge = project_boundary(half_car_width)
        
        # Track boundary positions
        left_track_boundary = project_boundary(-half_track)
        right_track_boundary = project_boundary(half_track)
        
        # Draw car boundaries (green if safe, red if near/over track edge)
        left_safe = self.state['lateral_offset'] > -half_track + half_car_width
        right_safe = self.state['lateral_offset'] < half_track - half_car_width
        
        if left_car_edge and left_car_edge[0] > 0 and left_car_edge[0] < width:
            color = (0, 255, 0) if left_safe else (255, 0, 0)
            pygame.draw.circle(canvas, color, (int(left_car_edge[0]), int(left_car_edge[1])), 8)
            pygame.draw.line(canvas, color, 
                           (left_car_edge[0], hood_top), 
                           (left_car_edge[0], left_car_edge[1]), 2)
        
        if right_car_edge and right_car_edge[0] > 0 and right_car_edge[0] < width:
            color = (0, 255, 0) if right_safe else (255, 0, 0)
            pygame.draw.circle(canvas, color, (int(right_car_edge[0]), int(right_car_edge[1])), 8)
            pygame.draw.line(canvas, color, 
                           (right_car_edge[0], hood_top), 
                           (right_car_edge[0], right_car_edge[1]), 2)
    
    def _draw_hud(self, canvas):
        """Draw HUD with telemetry."""
        if self.font is None:
            return
        
        # Create HUD info
        info_lines = [
            f"Speed: {self.state['speed']:.1f} m/s",
            f"Offset: {self.state['lateral_offset']:.2f} m",
            f"Heading: {np.rad2deg(self.state['heading']):.1f}°",
            f"Track Width: {self.current_track_width:.2f} m",
            f"Distance: {self.state['distance']:.1f} m",
        ]
        
        y_offset = 10
        for line in info_lines:
            text = self.font.render(line, True, (255, 255, 0))
            # Shadow
            canvas.blit(self.font.render(line, True, (0, 0, 0)), (11, y_offset + 1))
            # Text
            canvas.blit(text, (10, y_offset))
            y_offset += 30
    
    def _render_camera_view(self):
        """Render what Rover's camera actually sees (clean: sky, ground, white ropes only)."""
        # Create smaller surface for camera view
        cam_surface = pygame.Surface((640, 480))
        
        # Draw clean first-person view (no depth bands, no hood, no indicators)
        self._draw_first_person_view(cam_surface)
        
        # Convert to numpy array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(cam_surface)), axes=(1, 0, 2)
        )
    
    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
