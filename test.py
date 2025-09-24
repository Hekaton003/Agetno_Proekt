import time
import highway_env
import gymnasium
from stable_baselines3 import PPO, SAC, TD3
from training import config_racetrack, config_intersection
from myenv.continuous_roundabout import ContinuousRoundaboutEnv


# Helper function to create parallel environments
def make_env(env_id, config, render_mode=None):
    return gymnasium.make(env_id, config=config, render_mode=render_mode)


def test_model(model_path, model_class, iterations, config=None):
    print(f"Testing model from {model_path}...")
    env = ContinuousRoundaboutEnv()
    model = model_class.load(model_path)
    total_reward = 0
    for i in range(iterations):
        episode_reward = 0
        obs, info = env.reset()
        print(f'Iteration: {i}')
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_reward += episode_reward

    avg_reward = total_reward / iterations
    print(f"\n=== Evaluation Result ===")
    print(f"Average reward with {model_class} over {iterations} iterations: {avg_reward:.2f}")
    print('----')


def test_render_model(model_path, model_class, config=None, test_duration=30):
    print(f"Testing model from {model_path}...")
    env = ContinuousRoundaboutEnv()
    model = model_class.load(model_path)
    start_time = time.time()
    while time.time() - start_time < test_duration:
        done = False
        obs, info = env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()
        env.close()


if __name__ == "__main__":
    # --- Test Saved Model ---
    #test_render_model("roundabout/model_td3_MlpPolicy",TD3)
    # iterations = [50,100,150,200,250,300]
    iteration = int(input('Enter number of iterations: '))
    # Test PPO model
    test_model(f"roundabout/model_ppo_MlpPolicy_optimize", PPO, iteration)

    # Test SAC model
    test_model(f"roundabout/model_sac_MlpPolicy_optimize", SAC, iteration)

    # Test TD3 model
    test_model(f"roundabout/model_td3_MlpPolicy_optimize", TD3, iteration)

"""
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.ppo.ppo.PPO'> over 50 iterations: 67.54
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.ppo.ppo.PPO'> over 100 iterations: 60.45
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.ppo.ppo.PPO'> over 150 iterations: 63.19
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.ppo.ppo.PPO'> over 200 iterations: 67.33
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.ppo.ppo.PPO'> over 250 iterations: 62.88
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.ppo.ppo.PPO'> over 300 iterations: 67.53
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.sac.sac.SAC'> over 50 iterations: 75.75
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.sac.sac.SAC'> over 100 iterations: 70.53
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.sac.sac.SAC'> over 150 iterations: 70.37
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.sac.sac.SAC'> over 200 iterations: 73.57
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.sac.sac.SAC'> over 250 iterations: 71.30
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.sac.sac.SAC'> over 300 iterations: 73.20
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.td3.td3.TD3'> over 50 iterations: 4.28
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.td3.td3.TD3'> over 100 iterations: 3.93
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.td3.td3.TD3'> over 150 iterations: 1.41
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.td3.td3.TD3'> over 200 iterations: 4.40
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.td3.td3.TD3'> over 250 iterations: 2.42
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.td3.td3.TD3'> over 300 iterations: 0.84
----
"""

"""
Model with optimized parameters:
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.ppo.ppo.PPO'> over 50 iterations: 79.87
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.ppo.ppo.PPO'> over 100 iterations: 75.54
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.ppo.ppo.PPO'> over 150 iterations: 78.97
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.ppo.ppo.PPO'> over 200 iterations: 74.90
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.ppo.ppo.PPO'> over 250 iterations: 72.87
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.ppo.ppo.PPO'> over 300 iterations: 75.20
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.sac.sac.SAC'> over 50 iterations: 74.14
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.sac.sac.SAC'> over 100 iterations: 71.02
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.sac.sac.SAC'> over 150 iterations: 73.91
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.sac.sac.SAC'> over 200 iterations: 72.78
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.sac.sac.SAC'> over 250 iterations: 74.32
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.sac.sac.SAC'> over 300 iterations: 74.39
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.td3.td3.TD3'> over 50 iterations: 30.74
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.td3.td3.TD3'> over 100 iterations: 32.83
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.td3.td3.TD3'> over 150 iterations: 32.14
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.td3.td3.TD3'> over 200 iterations: 32.53
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.td3.td3.TD3'> over 250 iterations: 32.68
----
=== Evaluation Result ===
Average reward with <class 'stable_baselines3.td3.td3.TD3'> over 300 iterations: 32.28
----

"""
