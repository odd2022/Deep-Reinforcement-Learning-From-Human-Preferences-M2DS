import ale_py
print(ale_py.__version__)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Créer un environnement
def make_env():
    env = gym.make("Pong-v4")
    env = AtariWrapper(env)
    return env

env = DummyVecEnv([make_env])  
env = VecFrameStack(env, n_stack=4)  

# Définir le modèle PPO
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=2.5e-4,  
    batch_size=64,  
    n_steps=128,  
    verbose=1,
    tensorboard_log="./tensorboard_logs/ppo_logs/",  
)

# Entraîner le modèle
model.learn(total_timesteps=50_000_000)

# Sauvegarder l'agent
model.save("ppo_pong")
print("Modèle sauvegardé !")