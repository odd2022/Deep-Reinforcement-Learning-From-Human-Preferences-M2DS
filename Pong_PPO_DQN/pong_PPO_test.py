import ale_py
print(ale_py.__version__)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
import time

# Charger l'environnement
def make_env():
    env = gym.make("Pong-v4", render_mode="human")  
    env = AtariWrapper(env) 
    return env

# Créer l'environnement 
env = DummyVecEnv([make_env])  
env = VecFrameStack(env, n_stack=4)  

# Charger le modèle entraîné
model = PPO.load("ppo_pong")

# Tester l'agent
obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)  # L'agent choisit la meilleure action
    obs, reward, done, info = env.step(action)
    env.render()  # Affichage du jeu
    time.sleep(0.05)

env.close()
