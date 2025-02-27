import gymnasium as gym
import os
import ale_py
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor

# Dossier pour les logs TensorBoard
log_dir = "./tensorboard_logs/a2c_pong"
os.makedirs(log_dir, exist_ok=True)

# Fonction pour créer l'environnement avec `Monitor`
def make_env():
    env = gym.make("Pong-v4")
    env = AtariWrapper(env)  # Prétraitement d'Atari (grayscale, resize)
    env = Monitor(env, log_dir)  # Enregistre les rewards pour TensorBoard
    return env

# Création de l'environnement avec `VecFrameStack` et `VecNormalize`
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)  # Met les canaux en premier
env = VecTransposeImage(env)
# env = VecNormalize(env, norm_obs=True, norm_reward=True)  # Normalisation des rewards activée

obs = env.reset()
print("Shape of observations:", obs.shape)

# Définition du modèle A2C
model = A2C(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    n_steps=5,
    gamma=0.99,  
    ent_coef=0.01,  
    vf_coef=0.5,  
    max_grad_norm=0.5,  
    verbose=1,
    tensorboard_log=log_dir,
    # policy_kwargs={"normalize_images": False},  # Evite la double normalisation
    device="auto"  # Utilise le GPU si disponible
)

# Entraîner le modèle avec TensorBoard activé
model.learn(total_timesteps=5_000_000)

# Sauvegarde du modèle
model.save("a2c_pong")
print("Modèle A2C sauvegardé !")
