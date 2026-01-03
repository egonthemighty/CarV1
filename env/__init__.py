"""
CarV1 - Custom Gymnasium Environment Package
"""

from gymnasium.envs.registration import register

register(
    id='CarV1-v0',
    entry_point='env.car_env:CarEnv',
    max_episode_steps=1000,
)
