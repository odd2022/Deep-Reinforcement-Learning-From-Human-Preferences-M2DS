import time
import ale_py
print(ale_py.__version__)
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Créer un environnement Gym normal
def make_env():
    env = gym.make("Pong-v4", render_mode="human")  
    env = AtariWrapper(env)  # normalise et réduit l'image à 84x84
    return env

# Transformer l'environnement en environnement vectorisé 
env = DummyVecEnv([make_env])  
env = VecFrameStack(env, n_stack=4)  

# Charger le modèle entraîné
model = DQN.load("dqn_pong")

# Tester l'agent
obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)  # L'agent choisit la meilleure action
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
