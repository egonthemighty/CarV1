"""
Utility functions for the CarV1 project.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


def normalize_angle(angle):
    """Normalize angle to [-pi, pi] range."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def save_episode_data(episode_data, filename=None):
    """Save episode data to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{timestamp}.json"
    
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(episode_data, f, indent=2)
    
    return filepath


def plot_trajectory(positions, filename=None, title="Car Trajectory"):
    """Plot car trajectory and save to file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_{timestamp}.png"
    
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    
    positions = np.array(positions)
    
    plt.figure(figsize=(10, 8))
    plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1, alpha=0.7)
    plt.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    plt.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_rewards(rewards, filename=None, title="Episode Rewards"):
    """Plot rewards over time."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rewards_{timestamp}.png"
    
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, 'b-', linewidth=1, alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add moving average
    if len(rewards) > 10:
        window = min(50, len(rewards) // 10)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, 'r-', 
                linewidth=2, label=f'Moving Avg (window={window})')
        plt.legend()
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath


def create_logger(name, log_file=None):
    """Create a logger for debugging."""
    import logging
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        output_dir = Path("debug_output")
        output_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(output_dir / log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger
