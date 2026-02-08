# The Role of Deep Reinforcement Learning Algorithms in Autonomous Driving across Continuous Environments: Racetrack, Roundabout, and Intersection Tasks

## Overview 
This repository contains code for training and evaluating reinforcement learning models provided by the Stable Baselines3 library. The primary focus is on training models for racetrack-v0,intersection-v1 and roundabout-v0 environments using various DRL algorithms 
such as PPO, SAC, and TD3. For more details, see my [research paper](https://drive.google.com/file/d/1NTcBiVt9E1N-KrLUqvGOrhM_3aIlTXnu/view?usp=sharing)

## Project Structure
```                   
├── Results                        # A directory containing the evaluation results of PPO,SAC and TD3 for each environment
├── Visualizations                 # A directory that contains images about the visual description of the evaluation results of PPO,SAC and TD3 for each environment,
├── itersection                    # A directory containing the optimized and the default versions of the PPO,SAC and TD3 models about intersection-v1 environment along with their logs,
├── myenv
│   ├── continuous_roundabout.py   # A python file containing the ContinuousRoundaboutEnv class that modifies the original roundabout-v0 environment,
├── racetrack                      # A directory containing the optimized and the default versions of the PPO,SAC and TD3 models about racetrack-v0 environment  along with their logs
├── roundabout                     # A directory containing the optimized and the default versions of the PPO,SAC and TD3 models about roundabout-v0 environment along with their logs
├── hyper_optimization.py          # A python file containing the code for hyperparameter optimization of every algorithm
├── test.py                        # A python file containing the code for evaluate the models and their performances
├── training.py                    # A python file containing the code for training the models
├── visualization.py               # A python file containing the code for creating plots that visually describes the evaluation results of PPO,SAC and TD3 for each environment
├── README.md                      # Documentation
```
## Setup
1. Clone the repository
   ```sh
   git clone https://github.com/Hekaton003/Agetno_Proekt.git
   ```
2. Install dependencies like Optuna,highway-env,gymnasium and stable-baseline3
   
## Training models
```sh
python training.py
```
## Fine-Tuning the models
```sh
python hyper_optimization.py
```
## Testing models
```sh
python test.py
```
