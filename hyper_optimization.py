import gymnasium as gym
import highway_env
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
import optuna

from myenv.continuous_roundabout import ContinuousRoundaboutEnv
from training import OffroadTerminationWrapper, config_racetrack, config_intersection


def make_env(env_id, config=None):
    def _init():
        env = gym.make(env_id, config=config)
        env = OffroadTerminationWrapper(env)
        return env

    return _init


def make_env_single(env_id, config=None):
    def _init():
        env = gym.make(env_id, config=config)
        return env

    return _init


def objective(trial):
    # Hyperparameters tune
    learning_rate = trial.suggest_float('learning_rate', 3e-5, 3e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 200, 256])
    policy_kwargs = 'MlpPolicy'
    gamma = trial.suggest_float('gamma', 0.5, 0.9999)
    training_steps = 10000
    env = DummyVecEnv([lambda: ContinuousRoundaboutEnv()])
    model = TD3(
        env=env,
        policy=policy_kwargs,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        verbose=2,
        device="cpu",
    )
    model.learn(total_timesteps=training_steps)
    total_reward = 0.0
    eval_iterations = 50
    for i in range(eval_iterations):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            episode_reward += reward
            done = dones
        total_reward += episode_reward

    avg_reward = total_reward / eval_iterations
    return avg_reward


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    print("Best params:", study.best_params)
    print("Best average reward:", study.best_value)
