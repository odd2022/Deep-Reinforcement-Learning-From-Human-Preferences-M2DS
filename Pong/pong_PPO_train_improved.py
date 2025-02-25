import ale_py
print(ale_py.__version__)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.atari_wrappers import AtariWrapper
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter(log_dir="./tensorboard_logs/ppo_logs")

# Créer un environnement avec redimensionnement des images (84x84)
def make_env():
    env = gym.make("Pong-v4")
    env = AtariWrapper(env)  # <-- Normalise et réduit l'image à 84x84
    return env

env = DummyVecEnv([make_env])  # Vectorisation requise pour SB3
env = VecNormalize(env, norm_reward=True)
env = VecFrameStack(env, n_stack=4)  # Empilement des images

# Définir le modèle PPO
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=2.5e-4,  # Can be reduced for stability (e.g., 1e-4)
    batch_size=256,  # Larger batch sizes improve training stability
    n_steps=512,  # Increase for better updates (Pong is stable with ~512)
    gamma=0.99,  # Standard discount factor
    gae_lambda=0.95,  # Generalized Advantage Estimation (helps PPO)
    clip_range=0.1,  # Reduce clipping for more stable learning
    ent_coef=0.01,  # Increase entropy coefficient to encourage exploration
    vf_coef=0.5,  # Adjust value function loss scaling
    max_grad_norm=0.5,  # Prevent exploding gradients
    verbose=1,
    tensorboard_log="./tensorboard_logs/ppo_logs/"
)

# Entraîner le modèle
model.learn(total_timesteps=50_000_000)

# Sauvegarder l'agent
model.save("ppo_pong")
print("Modèle sauvegardé !")