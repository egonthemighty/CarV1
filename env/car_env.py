"""
Custom Gymnasium Environment for Self-Driving Car

This module implements a custom gymnasium environment where an agent learns
to drive a car in a simulated environment.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class CarEnv(gym.Env):
    """
    Custom Environment for training a self-driving car agent.
    
    Observation Space:
        - Car position (x, y)
        - Car velocity (vx, vy)
        - Car heading angle
        - Sensor readings (distance to obstacles)
        
    Action Space:
        - Steering angle: continuous [-1, 1]
        - Throttle: continuous [-1, 1] (negative for braking)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None, track_complexity=1):
        super().__init__()
        
        self.render_mode = render_mode
        self.track_complexity = track_complexity
        
        # Environment parameters
        self.window_width = 800
        self.window_height = 600
        self.max_steps = 1000
        self.dt = 0.1  # Time step
        
        # Car parameters
        self.car_length = 40
        self.car_width = 20
        self.max_speed = 100
        self.max_steering_angle = np.pi / 4
        
        # Number of distance sensors
        self.num_sensors = 8
        self.sensor_range = 200
        
        # Observation space: [x, y, vx, vy, angle, angular_velocity, sensor_1, ..., sensor_n]
        obs_low = np.array([
            0, 0,  # position
            -self.max_speed, -self.max_speed,  # velocity
            -np.pi, -np.pi,  # angle and angular velocity
        ] + [0] * self.num_sensors, dtype=np.float32)
        
        obs_high = np.array([
            self.window_width, self.window_height,  # position
            self.max_speed, self.max_speed,  # velocity
            np.pi, np.pi,  # angle and angular velocity
        ] + [self.sensor_range] * self.num_sensors, dtype=np.float32)
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
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
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize car state: [x, y, vx, vy, angle, angular_velocity]
        self.state = {
            'x': self.window_width / 2,
            'y': self.window_height / 2,
            'vx': 0,
            'vy': 0,
            'angle': 0,
            'angular_velocity': 0,
        }
        
        self.steps = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment."""
        steering, throttle = action
        
        # Clip actions to valid range
        steering = np.clip(steering, -1, 1)
        throttle = np.clip(throttle, -1, 1)
        
        # Update car physics
        self._update_physics(steering, throttle)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.steps >= self.max_steps
        
        info = self._get_info()
        
        self.steps += 1
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info
    
    def _update_physics(self, steering, throttle):
        """Update car physics based on actions."""
        # Convert steering to actual angle
        steering_angle = steering * self.max_steering_angle
        
        # Apply throttle (acceleration)
        acceleration = throttle * 50  # max acceleration
        
        # Update velocity
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
        
        # Keep angle in [-pi, pi]
        self.state['angle'] = np.arctan2(np.sin(self.state['angle']), 
                                         np.cos(self.state['angle']))
        
        # Update position
        self.state['x'] += self.state['vx'] * self.dt
        self.state['y'] += self.state['vy'] * self.dt
    
    def _get_sensor_readings(self):
        """Get distance sensor readings around the car."""
        sensors = []
        
        for i in range(self.num_sensors):
            angle = self.state['angle'] + (2 * np.pi * i / self.num_sensors)
            
            # Cast ray from car position
            distance = self._cast_ray(
                self.state['x'], 
                self.state['y'], 
                angle
            )
            sensors.append(distance)
        
        return np.array(sensors, dtype=np.float32)
    
    def _cast_ray(self, x, y, angle):
        """Cast a ray and return distance to nearest obstacle."""
        # Simple implementation: check distance to walls
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Distance to walls
        if dx > 0:
            dist_x = (self.window_width - x) / dx
        elif dx < 0:
            dist_x = -x / dx
        else:
            dist_x = self.sensor_range
        
        if dy > 0:
            dist_y = (self.window_height - y) / dy
        elif dy < 0:
            dist_y = -y / dy
        else:
            dist_y = self.sensor_range
        
        distance = min(dist_x, dist_y, self.sensor_range)
        
        return distance
    
    def _get_observation(self):
        """Construct observation from current state."""
        sensor_readings = self._get_sensor_readings()
        
        obs = np.array([
            self.state['x'],
            self.state['y'],
            self.state['vx'],
            self.state['vy'],
            self.state['angle'],
            self.state['angular_velocity'],
        ] + sensor_readings.tolist(), dtype=np.float32)
        
        return obs
    
    def _calculate_reward(self):
        """Calculate reward for current state."""
        reward = 0.0
        
        # Reward for forward velocity
        speed = np.sqrt(self.state['vx']**2 + self.state['vy']**2)
        reward += speed * 0.01
        
        # Penalty for being too close to walls
        min_sensor_reading = min(self._get_sensor_readings())
        if min_sensor_reading < 50:
            reward -= (50 - min_sensor_reading) * 0.01
        
        # Penalty for collision
        if self._is_collision():
            reward -= 100
        
        # Small time penalty to encourage efficiency
        reward -= 0.1
        
        return reward
    
    def _is_collision(self):
        """Check if car has collided with walls."""
        return bool(self.state['x'] < 0 or 
                    self.state['x'] > self.window_width or
                    self.state['y'] < 0 or 
                    self.state['y'] > self.window_height)
    
    def _is_terminated(self):
        """Check if episode should terminate."""
        return bool(self._is_collision())
    
    def _get_info(self):
        """Return additional information about current state."""
        return {
            'speed': np.sqrt(self.state['vx']**2 + self.state['vy']**2),
            'position': (self.state['x'], self.state['y']),
            'steps': self.steps,
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
    
    def _render_frame(self):
        """Render a single frame."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))
        
        # Draw car
        car_color = (0, 0, 255)
        car_rect = pygame.Rect(0, 0, self.car_length, self.car_width)
        car_surf = pygame.Surface((self.car_length, self.car_width), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, car_color, car_rect)
        
        # Rotate and position car
        rotated_car = pygame.transform.rotate(car_surf, -np.degrees(self.state['angle']))
        car_rect = rotated_car.get_rect(center=(self.state['x'], self.state['y']))
        canvas.blit(rotated_car, car_rect)
        
        # Draw sensors
        sensor_readings = self._get_sensor_readings()
        for i, distance in enumerate(sensor_readings):
            angle = self.state['angle'] + (2 * np.pi * i / self.num_sensors)
            end_x = self.state['x'] + distance * np.cos(angle)
            end_y = self.state['y'] + distance * np.sin(angle)
            pygame.draw.line(canvas, (200, 200, 200), 
                           (self.state['x'], self.state['y']), 
                           (end_x, end_y), 1)
        
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
