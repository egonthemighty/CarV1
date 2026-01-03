"""
Camera-based Line Following Environment for Self-Driving Car

Simple environment where car learns to follow two white lines (ropes)
using camera observations.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import cv2


class CameraLineFollowEnv(gym.Env):
    """
    Environment for camera-based line following.
    
    The car must stay between two white lines using visual input.
    
    Observation Space:
        Option 1 (Features): [left_line_pos, right_line_pos, car_angle, speed]
        Option 2 (Raw Image): Camera image array
        
    Action Space:
        - Steering: continuous [-1, 1]
        - Throttle: continuous [-1, 1]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        render_mode=None,
        use_raw_pixels=False,
        camera_resolution=(84, 84),
        track_width=60,  # cm, distance between lines
        car_width=30,  # cm
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.use_raw_pixels = use_raw_pixels
        self.camera_resolution = camera_resolution
        
        # Track parameters (in cm)
        self.track_width = track_width
        self.car_width = car_width
        
        # Rendering parameters (pixels)
        self.window_width = 800
        self.window_height = 600
        self.pixels_per_cm = 5  # Scale factor
        
        # Car parameters
        self.car_length_cm = 50  # 1:8 scale car
        self.car_length_px = int(self.car_length_cm * self.pixels_per_cm)
        self.car_width_px = int(self.car_width * self.pixels_per_cm)
        self.max_speed = 100  # cm/s
        self.max_steering_angle = np.pi / 4
        self.dt = 0.1
        
        # Track parameters
        self.line_thickness = 5  # pixels for rope
        self.track_length = 200  # meters (for progress tracking)
        
        # Episode parameters
        self.max_steps = 2000
        
        # Define observation space
        if use_raw_pixels:
            # Raw camera image
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(camera_resolution[0], camera_resolution[1], 1),
                dtype=np.uint8
            )
        else:
            # Extracted features: [left_line_offset, right_line_offset, heading_angle, speed]
            self.observation_space = spaces.Box(
                low=np.array([-track_width, -track_width, -np.pi, 0], dtype=np.float32),
                high=np.array([track_width, track_width, np.pi, self.max_speed], dtype=np.float32),
                dtype=np.float32
            )
        
        # Action space: [steering, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Pygame initialization
        self.window = None
        self.clock = None
        
        # State variables
        self.state = None
        self.steps = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize car in center of track
        self.state = {
            'x': self.window_width / 2,  # pixels, centered
            'y': self.window_height / 2,  # pixels, centered
            'vx': 0,  # cm/s
            'vy': 0,  # cm/s
            'angle': -np.pi / 2,  # pointing up
            'angular_velocity': 0,
            'lateral_offset': 0,  # cm from center, positive = right
        }
        
        self.steps = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action):
        """Execute one step."""
        steering, throttle = action
        
        # Clip actions
        steering = np.clip(steering, -1, 1)
        throttle = np.clip(throttle, -1, 1)
        
        # Update physics
        self._update_physics(steering, throttle)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.steps >= self.max_steps
        
        info = self._get_info()
        
        self.steps += 1
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info
    
    def _update_physics(self, steering, throttle):
        """Update car physics."""
        # Convert steering to angle
        steering_angle = steering * self.max_steering_angle
        
        # Apply throttle (acceleration in cm/s^2)
        acceleration = throttle * 200
        
        # Update velocity
        speed = np.sqrt(self.state['vx']**2 + self.state['vy']**2)
        
        # Forward direction
        self.state['vx'] += acceleration * np.cos(self.state['angle']) * self.dt
        self.state['vy'] += acceleration * np.sin(self.state['angle']) * self.dt
        
        # Apply friction
        friction = 0.95
        self.state['vx'] *= friction
        self.state['vy'] *= friction
        
        # Limit speed
        speed = np.sqrt(self.state['vx']**2 + self.state['vy']**2)
        if speed > self.max_speed:
            self.state['vx'] *= self.max_speed / speed
            self.state['vy'] *= self.max_speed / speed
        
        # Update angular velocity based on steering and speed
        if abs(speed) > 1:
            self.state['angular_velocity'] = steering_angle * (speed / 20)
        else:
            self.state['angular_velocity'] *= 0.9
        
        # Update angle
        self.state['angle'] += self.state['angular_velocity'] * self.dt
        self.state['angle'] = np.arctan2(np.sin(self.state['angle']), 
                                         np.cos(self.state['angle']))
        
        # Update position
        self.state['x'] += (self.state['vx'] * self.pixels_per_cm / 100) * self.dt * 10
        self.state['y'] += (self.state['vy'] * self.pixels_per_cm / 100) * self.dt * 10
        
        # Calculate lateral offset from track center (in cm)
        center_x = self.window_width / 2
        offset_pixels = self.state['x'] - center_x
        self.state['lateral_offset'] = offset_pixels / self.pixels_per_cm
    
    def _get_line_positions(self):
        """Get left and right line positions relative to car center."""
        # Lines are at fixed positions (vertical, centered)
        center_x = self.window_width / 2
        track_width_px = self.track_width * self.pixels_per_cm
        
        left_line_x = center_x - track_width_px / 2
        right_line_x = center_x + track_width_px / 2
        
        # Calculate offsets from car position (in cm)
        car_x = self.state['x']
        left_offset_cm = (left_line_x - car_x) / self.pixels_per_cm
        right_offset_cm = (right_line_x - car_x) / self.pixels_per_cm
        
        return left_offset_cm, right_offset_cm
    
    def _get_observation(self):
        """Get observation based on mode."""
        if self.use_raw_pixels:
            return self._get_camera_image()
        else:
            return self._get_feature_observation()
    
    def _get_feature_observation(self):
        """Extract features from scene."""
        left_offset, right_offset = self._get_line_positions()
        
        # Angle relative to track (track goes straight up, so relative to -pi/2)
        heading_error = self.state['angle'] - (-np.pi / 2)
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Speed
        speed = np.sqrt(self.state['vx']**2 + self.state['vy']**2)
        
        obs = np.array([
            left_offset,
            right_offset,
            heading_error,
            speed
        ], dtype=np.float32)
        
        return obs
    
    def _get_camera_image(self):
        """Get camera view as image array."""
        # Render scene
        surface = self._render_scene()
        
        # Extract camera view region (front of car)
        camera_view = self._extract_camera_view(surface)
        
        # Preprocess to target resolution
        camera_view = cv2.resize(camera_view, self.camera_resolution)
        
        # Convert to grayscale
        if len(camera_view.shape) == 3:
            camera_view = cv2.cvtColor(camera_view, cv2.COLOR_RGB2GRAY)
        
        # Add channel dimension
        camera_view = np.expand_dims(camera_view, axis=-1)
        
        return camera_view.astype(np.uint8)
    
    def _extract_camera_view(self, surface):
        """Extract camera view from rendered surface."""
        # Get region in front of car
        view_width = 200
        view_height = 150
        
        car_x = int(self.state['x'])
        car_y = int(self.state['y'])
        
        # Extract region ahead of car
        x1 = max(0, car_x - view_width // 2)
        x2 = min(self.window_width, car_x + view_width // 2)
        y1 = max(0, car_y - view_height)
        y2 = car_y
        
        # Get pixels
        view = pygame.surfarray.array3d(surface)
        view = view[x1:x2, y1:y2]
        view = np.transpose(view, (1, 0, 2))
        
        return view
    
    def _calculate_reward(self):
        """Calculate reward."""
        reward = 0.0
        
        # Get line positions
        left_offset, right_offset = self._get_line_positions()
        
        # Penalty for being off-center
        center_offset = abs(self.state['lateral_offset'])
        reward -= center_offset * 0.02
        
        # Check if out of bounds (crossed a line)
        half_car_width = self.car_width / 2
        left_boundary = -self.track_width / 2 + half_car_width
        right_boundary = self.track_width / 2 - half_car_width
        
        if self.state['lateral_offset'] < left_boundary or self.state['lateral_offset'] > right_boundary:
            # Out of bounds
            reward -= 10.0
        else:
            # Staying on track bonus
            reward += 0.5
        
        # Reward for forward speed (when aligned)
        speed = np.sqrt(self.state['vx']**2 + self.state['vy']**2)
        heading_error = abs(self.state['angle'] - (-np.pi / 2))
        if heading_error < 0.3:  # Roughly aligned
            reward += speed * 0.01
        
        # Small time penalty
        reward -= 0.1
        
        return reward
    
    def _is_terminated(self):
        """Check if episode should terminate."""
        # Check if car went off track
        half_car_width = self.car_width / 2
        left_boundary = -self.track_width / 2 + half_car_width
        right_boundary = self.track_width / 2 - half_car_width
        
        off_track = (self.state['lateral_offset'] < left_boundary - 5 or 
                    self.state['lateral_offset'] > right_boundary + 5)
        
        # Check if car went off screen
        off_screen = (self.state['x'] < 50 or 
                     self.state['x'] > self.window_width - 50 or
                     self.state['y'] < 50 or 
                     self.state['y'] > self.window_height - 50)
        
        return bool(off_track or off_screen)
    
    def _get_info(self):
        """Return additional info."""
        speed = np.sqrt(self.state['vx']**2 + self.state['vy']**2)
        left_offset, right_offset = self._get_line_positions()
        
        return {
            'speed': speed,
            'lateral_offset': self.state['lateral_offset'],
            'left_line_offset': left_offset,
            'right_line_offset': right_offset,
            'steps': self.steps,
            'on_track': abs(self.state['lateral_offset']) < self.track_width / 2 - self.car_width / 2,
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
    
    def _render_scene(self):
        """Render the scene to a surface."""
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((60, 60, 60))  # Dark gray background
        
        # Draw white lines (ropes)
        center_x = self.window_width / 2
        track_width_px = self.track_width * self.pixels_per_cm
        
        left_line_x = int(center_x - track_width_px / 2)
        right_line_x = int(center_x + track_width_px / 2)
        
        # Draw lines as thick white lines
        pygame.draw.line(canvas, (255, 255, 255), 
                        (left_line_x, 0), (left_line_x, self.window_height), 
                        self.line_thickness)
        pygame.draw.line(canvas, (255, 255, 255), 
                        (right_line_x, 0), (right_line_x, self.window_height), 
                        self.line_thickness)
        
        # Draw car
        car_color = (0, 120, 255)
        car_surf = pygame.Surface((self.car_length_px, self.car_width_px), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, car_color, car_surf.get_rect())
        
        # Add front indicator
        pygame.draw.rect(car_surf, (255, 0, 0), 
                        (self.car_length_px - 10, 0, 10, self.car_width_px))
        
        # Rotate and position car
        rotated_car = pygame.transform.rotate(car_surf, -np.degrees(self.state['angle']))
        car_rect = rotated_car.get_rect(center=(self.state['x'], self.state['y']))
        canvas.blit(rotated_car, car_rect)
        
        # Draw info text
        if pygame.font.get_init():
            font = pygame.font.Font(None, 24)
            speed = np.sqrt(self.state['vx']**2 + self.state['vy']**2)
            info_text = f"Speed: {speed:.1f} cm/s  Offset: {self.state['lateral_offset']:.1f} cm  Step: {self.steps}"
            text_surf = font.render(info_text, True, (255, 255, 255))
            canvas.blit(text_surf, (10, 10))
        
        return canvas
    
    def _render_frame(self):
        """Render a single frame."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Camera Line Follow Environment")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = self._render_scene()
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
