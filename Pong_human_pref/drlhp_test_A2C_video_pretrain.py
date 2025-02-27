import numpy as np
import cv2
from collections import deque
from itertools import combinations
from random import shuffle
import os
import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from drlhp_train_A2C_video import HumanPreferencesEnvWrapper

# Nom du fichier pour sauvegarder le modèle de récompense
REWARD_MODEL_PATH = "reward_predictor.pth"

# Créer l'environnement
env = DummyVecEnv([lambda: HumanPreferencesEnvWrapper(
    gym.make("Pong-v4"), segment_length=20, synthetic_prefs=False, log_dir="logs/"
)])

# Collecter des segments initiaux pour le pré-entraînement
def collect_initial_segments(env, num_steps=1000):
    """Permet de collecter des segments initiaux en jouant de manière aléatoire."""
    print(f"Collecte de {num_steps} étapes de jeu pour initialiser les segments...")

    obs = env.reset()
    for _ in range(num_steps):
        action = np.array([env.action_space.sample()])  # Assurer que c'est un tableau NumPy
        env.step(action)  # Exécute l'action et collecte les données

    print(f"{num_steps} étapes collectées. Prêt pour le pré-entraînement !")


# Charger le modèle de prédicteur de récompense s'il existe
def load_reward_predictor(env, model_path=REWARD_MODEL_PATH):
    if os.path.exists(model_path):
        env.get_attr("reward_predictor")[0].load_state_dict(torch.load(model_path))
        print(f"Modèle pré-entraîné chargé depuis {model_path}")
    else:
        print("Aucun modèle pré-entraîné trouvé, entraînement depuis zéro.")

# Phase de pré-entraînement du prédicteur de récompense
def pretrain_reward_predictor(env, num_pretrain_steps=10):
    print("Début du pré-entraînement du prédicteur de récompense...")

    pref_interface = env.get_attr("pref_interface")[0]
    reward_predictor = env.get_attr("reward_predictor")[0]
    optimizer = env.get_attr("optimizer")[0]
    criterion = env.get_attr("criterion")[0]

    for step in range(num_pretrain_steps):
        pref_result = pref_interface.query_user()
        if pref_result:
            s1, s2, preference = pref_result
            loss = reward_predictor.train_model(s1, s2, preference, optimizer, criterion)
            print(f"Pré-entraînement Step {step + 1}/{num_pretrain_steps} - Loss: {loss}")

    print("Pré-entraînement terminé !")

    # Sauvegarde du modèle pré-entraîné
    torch.save(reward_predictor.state_dict(), REWARD_MODEL_PATH)
    print(f"Modèle pré-entraîné sauvegardé sous {REWARD_MODEL_PATH}")

# Charger un modèle existant si disponible
load_reward_predictor(env)

# Effectuer le pré-entraînement avant l'entraînement du modèle A2C
# Avant le pré-entraînement, on collecte des données
collect_initial_segments(env, num_steps=1000)
pretrain_reward_predictor(env, num_pretrain_steps=10)

# Création du modèle A2C
model = A2C("MlpPolicy", env, verbose=1)

# Nombre d'épisodes d'entraînement
num_episodes = 50

# Boucle d'entraînement principale
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        step_result = env.step(action)

        if len(action) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated, truncated = done, False

    # Passage aux récompenses apprises à mi-chemin
    if episode == num_episodes // 2:
        env.get_attr("switch_to_predicted_reward")[0]()
        print("Passage aux récompenses prédites à mi-entraînement.")

    # Comparaison et apprentissage du prédicteur de récompense
    pref_interface = env.get_attr("pref_interface")[0]
    pref_result = pref_interface.query_user()

    if pref_result:
        s1, s2, preference = pref_result
        loss = env.get_attr("reward_predictor")[0].train_model(
            s1, s2, preference, env.get_attr("optimizer")[0], env.get_attr("criterion")[0]
        )
        print(f"Loss du modèle de récompense : {loss}")

    model.learn(total_timesteps=500)

print("Entraînement terminé avec A2C !")
