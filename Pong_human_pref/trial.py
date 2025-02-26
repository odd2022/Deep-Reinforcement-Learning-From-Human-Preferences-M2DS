import gymnasium as gym
import numpy as np
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time

class PrefInterface:
    def __init__(self, max_segments=100):
        self.max_segments = max_segments
        self.segment_queue = mp.Queue()  # Utilisation d'une queue partagée

    def add_segment(self, segment):
        if self.segment_queue.qsize() >= self.max_segments:
            self.segment_queue.get()  # Supprime l'ancien segment pour éviter d'encombrer la queue
        self.segment_queue.put(segment)
        print(f"✅ Segment ajouté ! Total segments en mémoire: {self.segment_queue.qsize()}")

    def query_user(self):
        if self.segment_queue.qsize() < 2:
            print("⚠️ Pas assez de segments pour une comparaison.")
            return None

        # Récupération des 2 derniers segments
        s1 = self.segment_queue.get()
        s2 = self.segment_queue.get()

        print(f"📢 **NOUVELLE COMPARAISON** ({self.segment_queue.qsize()} segments restants)")
        print("1: Segment 1")
        print("2: Segment 2")
        choice = input("Quel segment préférez-vous ? (1/2) ou (Enter pour passer) : ")

        if choice in ["1", "2"]:
            print(f"🎯 Préférence choisie : {choice}")
            return (s1, s2, (1.0, 0.0) if choice == "1" else (0.0, 1.0))

        print("⚠️ Aucune préférence donnée, on ignore cette comparaison.")
        return None


class RewardPredictor(nn.Module):
    def __init__(self, input_dim):
        super(RewardPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
    def train_model(self, s1, s2, preference, optimizer, criterion):
        optimizer.zero_grad()

        # Vérifier que les segments ne sont pas vides
        if not s1 or not s2:
            print("🚨 Erreur: Un des segments est vide, impossible d'entraîner le modèle.")
            return None

        # Extraire uniquement les observations
        s1_obs = np.array([obs for obs, _ in s1], dtype=np.float32)
        s2_obs = np.array([obs for obs, _ in s2], dtype=np.float32)

        # Vérifier que les tableaux ne sont pas vides après extraction
        if s1_obs.size == 0 or s2_obs.size == 0:
            print("🚨 Erreur: Un des segments est vide après extraction des observations.")
            return None

        # Calculer la moyenne des observations sur le segment (assurer une forme correcte)
        try:
            s1_tensor = torch.tensor(np.mean(s1_obs, axis=0), dtype=torch.float32).unsqueeze(0)
            s2_tensor = torch.tensor(np.mean(s2_obs, axis=0), dtype=torch.float32).unsqueeze(0)
        except ValueError as e:
            print(f"🚨 Erreur lors du calcul de la moyenne des segments: {e}")
            return None

        # Vérification des dimensions
        if s1_tensor.shape != s2_tensor.shape:
            print(f"🚨 Erreur: Formes incohérentes entre s1 ({s1_tensor.shape}) et s2 ({s2_tensor.shape})")
            return None

        # Prédiction des récompenses
        r1 = self.forward(s1_tensor).squeeze()
        r2 = self.forward(s2_tensor).squeeze()

        # Assurer que r1 et r2 ont la bonne dimension
        if r1.dim() == 0:
            r1 = r1.unsqueeze(0)
        if r2.dim() == 0:
            r2 = r2.unsqueeze(0)

        # Préparation des entrées pour la loss
        input_tensor = torch.sigmoid(r1 - r2).unsqueeze(0)  # Assurer une dimension correcte
        target = torch.tensor([preference[0]], dtype=torch.float32)

        # Vérification des dimensions avant de calculer la loss
        if input_tensor.shape != target.shape:
            print(f"🚨 Correction: Reshape input_tensor ({input_tensor.shape}) -> ({target.shape})")
            input_tensor = input_tensor.view_as(target)

        # Calcul de la loss
        loss = criterion(input_tensor, target)
        loss.backward()
        optimizer.step()

        print(f"✅ Entraînement terminé, Loss: {loss.item()}")
        return loss.item()



class HumanPreferencesEnvWrapper(gym.Wrapper):
    def __init__(self, env, segment_length=20, max_segments=1000):
        super(HumanPreferencesEnvWrapper, self).__init__(env)
        self.segment_length = segment_length
        self.segments = deque(maxlen=max_segments)
        self.current_segment = []
        self.pref_interface = PrefInterface()
        self.reward_predictor = RewardPredictor(input_dim=env.observation_space.shape[0])
        self.use_learned_reward = False
        self.optimizer = optim.Adam(self.reward_predictor.parameters(), lr=1e-3)
        self.criterion = nn.BCEWithLogitsLoss()  # Mieux adapté aux préférences
        self.pref_queue = mp.Queue()  # Communication entre processus


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated 
        self.current_segment.append((obs, reward))

        if len(self.current_segment) >= self.segment_length:
            self._store_segment()
            self.current_segment = []

        if self.use_learned_reward:
            reward = self.reward_predictor(torch.tensor(obs, dtype=torch.float32)).item()
            print(f"📢 Récompense prédite : {reward}")  # Debugging

        if done:
            print(f"Épisode terminé avec un score de {reward}")

        return obs, reward, done, info

        
    def _store_segment(self):
        segment = list(self.current_segment)
        self.segments.append(segment)
        self.pref_interface.add_segment(segment)
    
    def train_reward_predictor(self):
        """Entraîner le modèle de récompense basé sur les préférences collectées."""
        while True:
            print("En attente d'une préférence utilisateur...")
            pref = self.pref_interface.query_user()
            if pref:
                print("Préférence reçue, entraînement en cours...")
                s1, s2, preference = pref
                loss = self.reward_predictor.train_model(s1, s2, preference, self.optimizer, self.criterion)
                print(f"Loss: {loss}")
            print("Attente de nouvelles préférences...")
            time.sleep(1)  # Petit délai pour éviter une boucle infinie trop rapide

    def switch_to_predicted_reward(self):
        """Activer l'utilisation de la récompense prédite."""
        self.use_learned_reward = True
