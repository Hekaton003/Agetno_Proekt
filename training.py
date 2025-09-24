import gymnasium as gym
import highway_env
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from myenv.continuous_roundabout import ContinuousRoundaboutEnv

envIDs = ['racetrack-v0', 'intersection-v1', 'ContinuousRoundabout-v0']
config_racetrack = {
    "other_vehicles": 4,
    "collision_reward": -5,
    "lane_centering_reward": 3
}
config_intersection = {
    "offroad_terminal": True,
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True
    },
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "absolute": True,
        "flatten": True,
        "observe_intentions": False,
    },
    "collision_reward": -50,
    "high_speed_reward": 10,
    "arrived_reward": 20
}


class OffroadTerminationWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # If the agent is off-road
        if not self.unwrapped.vehicle.on_road:
            terminated = True
            reward = -10

        return obs, reward, terminated, truncated, info


def make_env(env_id, config=None):
    def _init():
        if env_id == 'racetrack-v0':
            env = gym.make(env_id, config=config)
            env = OffroadTerminationWrapper(env)
            return env
        elif env_id == 'intersection-v1':
            env = gym.make(env_id, config=config)
            return env
        else:
            env = gym.make(env_id, config=config)
            return env

    return _init


def optimize_models(policy, env, training_steps):
    print("PPO")
    # PPO RL model
    model_ppo = PPO(policy, env,
                    learning_rate=0.00030548161744990866,
                    batch_size=128,
                    gamma=0.9278522727265088,
                    tensorboard_log=f"roundabout/",
                    verbose=2,
                    device="cpu",
                    )
    model_ppo.learn(int(training_steps))
    model_ppo.save(f"roundabout/model_ppo_{policy}_optimize")
    print("SAC")
    # SAC RL model
    model_sac = SAC(policy, env, learning_rate=0.0009124459816241446, batch_size=256, gamma=0.9611111561070567,
                    tensorboard_log=f"roundabout/",
                    verbose=2, device="cpu")
    model_sac.learn(int(training_steps), log_interval=4)
    model_sac.save(f"roundabout/model_sac_{policy}_optimize")

    # TD3 RL model
    model_td3 = TD3(policy, env, learning_rate=0.002584655518886114, batch_size=128, gamma=0.8656588059461084,
                    tensorboard_log=f"roundabout/",
                    verbose=2, device="cpu")
    model_td3.learn(int(training_steps), log_interval=4)
    model_td3.save(f"roundabout/model_td3_{policy}_optimize")


def default_models(policy, env, training_steps):
    print("PPO")
    # PPO RL model
    model_ppo = PPO(policy, env,
                    learning_rate=3e-4,
                    batch_size=128,
                    gamma=0.99,
                    tensorboard_log=f"roundabout/",
                    verbose=2,
                    device="cpu",
                    )
    model_ppo.learn(int(training_steps))
    model_ppo.save(f"roundabout/model_ppo_{policy}")
    print("SAC")
    #SAC RL model
    model_sac = SAC(policy, env, learning_rate=3e-4, batch_size=128, gamma=0.99, tensorboard_log=f"roundabout/",
                    verbose=2, device="cpu")
    model_sac.learn(int(training_steps), log_interval=4)
    model_sac.save(f"roundabout/model_sac_{policy}")

    #TD3 RL model
    model_td3 = TD3(policy, env, learning_rate=3e-4, batch_size=128, gamma=0.99, tensorboard_log=f"roundabout/",
                    verbose=2, device="cpu")
    model_td3.learn(int(training_steps), log_interval=4)
    model_td3.save(f"roundabout/model_td3_{policy}")


if __name__ == '__main__':
    IsOptimize = int(input('Enter 1 for yes, 0 for no (optimize): '))
    num_envs = 4
    training_steps = 200000
    env = SubprocVecEnv([lambda: ContinuousRoundaboutEnv() for _ in range(num_envs)])
    policy = 'MlpPolicy'
    print(f'Action space: {env.action_space}')
    print(env.observation_space)
    if IsOptimize:
        optimize_models(policy, env, training_steps)
    else:
        default_models(policy, env, training_steps)
    """
     # Action space: Box(-1.0, 1.0, (1,), float32)
    # continues action space with 1D vector space where it can have 3 values -1.0 (left), 0.0(straight) and 1.0 (right)
    print(f'Observation space: {env.observation_space}')
    # continues observation space with 3D vector space that has a range from -inf to +inf where the first number is
    # chanel that provides info about the cell and the last 2 cell represents the window size of the game
    """
