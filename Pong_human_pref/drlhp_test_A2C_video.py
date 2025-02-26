import numpy as np
import cv2
from collections import deque
from itertools import combinations
from random import shuffle
import ale_py
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from drlhp_train_A2C_video import HumanPreferencesEnvWrapper

env = DummyVecEnv([lambda: HumanPreferencesEnvWrapper(gym.make("Pong-v4"), segment_length=20, synthetic_prefs=False, log_dir="logs/")])
model = A2C("MlpPolicy", env, verbose=1)

num_episodes = 50

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        step_result = env.step(action)

        if len(action) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated, truncated = done, False

    if episode == num_episodes // 2:
        env.get_attr("switch_to_predicted_reward")[0]()

    pref_interface = env.get_attr("pref_interface")[0]
    pref_result = pref_interface.query_user()

    if pref_result:
        s1, s2, preference = pref_result
        loss = env.get_attr("reward_predictor")[0].train_model(
            s1, s2, preference, env.get_attr("optimizer")[0], env.get_attr("criterion")[0]
        )
        print(f"Loss du modèle de récompense : {loss}")

    model.learn(total_timesteps=500)

print("Entraînement terminé avec A2C !")
