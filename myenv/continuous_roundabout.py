from highway_env.envs import RoundaboutEnv
import numpy as np

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from highway_env.envs.roundabout_env import RoundaboutEnv


class ContinuousRoundaboutEnv(RoundaboutEnv):
    """
    Custom RoundaboutEnv with continuous action space for PPO, SAC, TD3.
    """
    metadata = {'render.modes': ['human','rgb_array']}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "absolute": True,
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-15, 15],
                        "vy": [-15, 15],
                    },
                },
                "action": {"type": "ContinuousAction"},
                "incoming_vehicle_destination": None,
                "collision_reward": -1.0,
                "on_road_reward": 1.0,
                "lane_center_reward": 0.3,
                "speed_reward": 0.2,
                "steering_penalty": -0.05,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "duration": 11,
                "normalize_reward": True,
            }
        )
        return config

    def __init__(self,render_mode=None, config=None):
        super().__init__(config)
        # continuous action: [acceleration, steering]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.render_mode = render_mode

    def _rewards(self, action) -> dict:
        """
        Return dict of component rewards.
        """
        # distance to lane center (y deviation in local coordinates)
        try:
            lane_coords = self.vehicle.lane.local_coordinates(self.vehicle.position)
            dist_center = abs(lane_coords[1])
        except Exception:
            dist_center = 0.0
        max_speed = 15.0
        rewards = {
            "collision_reward": -7.0 if self.vehicle.crashed else 3,
            "on_road_reward": 12.0 if self.vehicle.on_road else -10.0,
            "lane_center_reward": -dist_center,
            "speed_reward": self.vehicle.speed / (max_speed + 1e-6),
            "steering_penalty": -0.1 * abs(action[1]) if isinstance(action, (list, np.ndarray)) else 0.0,
        }
        return rewards

    def _reward(self, action) -> float:
        """
        Convert dict rewards to scalar float for SB3.
        """
        rewards = self._rewards(action)  # dict
        total = sum(self.config.get(name, 1.0) * r for name, r in rewards.items())
        return float(total)
