import ale_py
print(ale_py.__version__)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Créer un environnement avec redimensionnement des images (84x84)
def make_env():
    env = gym.make("Pong-v4")
    env = AtariWrapper(env)  # <-- Normalise et réduit l'image à 84x84
    return env

env = DummyVecEnv([make_env])  # Vectorisation requise pour SB3
env = VecFrameStack(env, n_stack=4)  # Empilement des images

# Définir le modèle PPO
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=2.5e-4,  # Apprentissage adapté à Pong
    batch_size=64,  # Taille du batch (plus grand que DQN)
    n_steps=128,  # Nombre de steps avant mise à jour
    verbose=1,
    tensorboard_log="./tensorboard_logs/ppo_logs/",  # Suivi de l'apprentissage
)

# Entraîner le modèle
model.learn(total_timesteps=50_000_000)

# Sauvegarder l'agent
model.save("ppo_pong")
print("Modèle sauvegardé !")