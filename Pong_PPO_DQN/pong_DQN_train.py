import ale_py
print(ale_py.__version__)
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Créer un environnement Pong avec vectorisation et stacking d'images
env = make_atari_env("Pong-v4", n_envs=1)

# Empiler 4 images pour donner une mémoire au modèle
env = VecFrameStack(env, n_stack=4)

# Définir le modèle DQN
model = DQN(
    "CnnPolicy", 
    env,
    learning_rate=1e-4,  
    buffer_size=100000,
    learning_starts=10000,  
    batch_size=32, 
    gamma=0.99,  
    target_update_interval=1000,  
    train_freq=4,  
    exploration_fraction=0.1,  
    exploration_final_eps=0.02,  
    verbose=1,  
    tensorboard_log="./dqn_logs/",
)

# Entraîner le modèle pendant 500 000 steps
model.learn(total_timesteps=500000)

model.save("dqn_pong")

