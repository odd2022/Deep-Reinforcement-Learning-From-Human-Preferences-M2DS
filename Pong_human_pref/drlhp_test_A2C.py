import time
import gymnasium as gym
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from drlhp_train_A2C import HumanPreferencesEnvWrapper

# Initialize environment inside DummyVecEnv
env = DummyVecEnv([lambda: HumanPreferencesEnvWrapper(gym.make("CartPole-v1"), segment_length=20)])

# Initialize A2C model
model = A2C("MlpPolicy", env, verbose=1, learning_rate=1e-3, gamma=0.99, n_steps=5)

num_episodes = 50

for episode in range(num_episodes):
    obs = env.reset()  # ✅ Now correctly receives a single value (NumPy array)
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)  # ✅ Now expecting only 4 values
        total_reward += reward

    print(f"🎯 Episode {episode + 1} completed with a score of {total_reward}")

    if episode == num_episodes // 2:
        env.get_attr("switch_to_predicted_reward")[0]()  # ✅ Call function inside VecEnv

    pref = env.get_attr("pref_interface")[0].query_user()
    if pref:
        s1, s2, preference = pref
        loss = env.get_attr("reward_predictor")[0].train_model(
            s1, s2, preference, env.get_attr("optimizer")[0], env.get_attr("criterion")[0]
        )
        print(f"Loss: {loss}")

    model.learn(total_timesteps=500)

print("Training completed with A2C!")
