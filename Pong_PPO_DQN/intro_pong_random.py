import gymnasium as gym
import ale_py
print(ale_py.__version__)
print(gym.envs.registry.keys())
from stable_baselines3 import DQN

# Créer l’environnement Pong
env = gym.make("Pong-v4", render_mode="human")

# Vérifier l’espace d’actions et l’espace d’observations
print("Espace d'observation :", env.observation_space)
print("Espace d'actions :", env.action_space)

# Tester une exécution aléatoire
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Action aléatoire
    obs, reward, done, truncated, info = env.step(action)
    env.render()

env.close()
