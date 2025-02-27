import ale_py
print(ale_py.__version__)
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Créer un environnement Pong avec vectorisation et stacking d'images
env = make_atari_env("Pong-v4", n_envs=1)
# env = gym.make("Pong-v4")
# Le transformer en environnement vectorisé (nécessaire pour Stable-Baselines3)
# env = DummyVecEnv([lambda: env])
# Empiler 4 images pour donner une mémoire au modèle
env = VecFrameStack(env, n_stack=4)

# Définir le modèle DQN
model = DQN(
    "CnnPolicy",  # Utilisation d’un CNN car l'entrée est une image
    env,
    learning_rate=1e-4,  # Taux d’apprentissage
    buffer_size=100000,  # Taille du buffer de replay
    learning_starts=10000,  # Nombre d'étapes avant d'apprendre
    batch_size=32,  # Taille du batch d'entraînement
    gamma=0.99,  # Facteur de discount
    target_update_interval=1000,  # Mise à jour du réseau cible
    train_freq=4,  # Fréquence des mises à jour
    exploration_fraction=0.1,  # Phase d'exploration (ε-greedy)
    exploration_final_eps=0.02,  # Valeur minimale de epsilon
    verbose=1,  # Affichage des logs
    tensorboard_log="./dqn_logs/",
)

# Entraîner le modèle pendant 500 000 steps (peut être ajusté)
model.learn(total_timesteps=500000)

# Sauvegarder l'agent entraîné
model.save("dqn_pong")

