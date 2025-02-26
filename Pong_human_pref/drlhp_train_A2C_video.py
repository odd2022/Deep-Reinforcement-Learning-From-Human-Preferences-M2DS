import numpy as np
import cv2
from collections import deque
from itertools import combinations
from random import shuffle

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class PrefInterface:
    def __init__(self, synthetic_prefs, max_segs, log_dir):
        self.synthetic_prefs = synthetic_prefs
        self.max_segs = max_segs
        self.segments = deque(maxlen=max_segs)  # Circular buffer
        self.tested_pairs = set()
        self.log_dir = log_dir

    def add_segment(self, segment):
        """Ajoute un segment à la liste et affiche son ajout."""
        self.segments.append(segment)
        print(f"📝 Nouveau segment ajouté ! Total : {len(self.segments)}")

    def sample_seg_pair(self):
        """Sélectionne deux segments non encore comparés."""
        if len(self.segments) < 2:
            print("⏳ Pas encore assez de segments pour comparaison.")
            return None  

        segment_idxs = list(range(len(self.segments)))
        shuffle(segment_idxs)
        possible_pairs = combinations(segment_idxs, 2)
        
        for i1, i2 in possible_pairs:
            s1, s2 = self.segments[i1], self.segments[i2]
            s1_frames = np.array([obs for obs, _ in s1])
            s2_frames = np.array([obs for obs, _ in s2])

            if isinstance(s1_frames, np.ndarray) and isinstance(s2_frames, np.ndarray):
                s1_hash = hash(s1_frames.tobytes())
                s2_hash = hash(s2_frames.tobytes())
            else:
                print("⚠️ Erreur de format de frame, attendu numpy.ndarray.")
                continue  

            if ((s1_hash, s2_hash) not in self.tested_pairs) and \
               ((s2_hash, s1_hash) not in self.tested_pairs):
                self.tested_pairs.add((s1_hash, s2_hash))
                self.tested_pairs.add((s2_hash, s1_hash))
                return s1, s2

        print("⏳ Pas encore assez de paires de segments non testées.")
        return None  

    def query_user(self):
        """Demande à l'utilisateur de choisir un segment préféré."""
        if len(self.segments) < 2:
            print("⏳ Pas encore assez de segments pour comparaison.")
            return None  

        pair = self.sample_seg_pair()  
        if pair:
            s1, s2 = pair
            return self.ask_user(s1, s2)
        else:
            print("⏳ Pas encore assez de segments pour comparaison.")
            return None

    def ask_user(self, s1, s2):
        """Affiche les segments côte à côte et demande un choix."""
        seg_len = len(s1)
        height, width = s1[0][0].shape[:2]

        border = np.zeros((height, 10), dtype=np.uint8)

        for t in range(seg_len):
            frame_left = cv2.cvtColor(s1[t][0], cv2.COLOR_RGB2GRAY)  
            frame_right = cv2.cvtColor(s2[t][0], cv2.COLOR_RGB2GRAY)  
            combined_frame = np.hstack((frame_left, border, frame_right))

            cv2.imshow("Segment Comparison", combined_frame)
            cv2.waitKey(100)

        print("\n[PREFERENCE] : Tapez 'L' pour gauche, 'R' pour droite, 'E' pour neutre, 'Q' pour ignorer.")

        while True:
            key = cv2.waitKey(0) & 0xFF  

            if key == ord("l"):
                pref = (1.0, 0.0)  
                break
            elif key == ord("r"):
                pref = (0.0, 1.0)  
                break
            elif key == ord("e"):
                pref = (0.5, 0.5)  
                break
            elif key == ord("q"):
                pref = None  
                break
            else:
                print("⚠️ Touche invalide, veuillez appuyer sur L, R, E ou Q.")

        cv2.destroyAllWindows()
        return pref

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

class HumanPreferencesEnvWrapper(gym.Wrapper):
    def __init__(self, env, segment_length=20, max_segments=1000, synthetic_prefs=False, log_dir="./logs"):
        super().__init__(env)
        self.segment_length = segment_length
        self.current_segment = []
        self.segments = deque(maxlen=max_segments)
        self.pref_interface = PrefInterface(synthetic_prefs, max_segments, log_dir)
        self.reward_predictor = RewardPredictor(input_dim=env.observation_space.shape[0])
        self.optimizer = optim.Adam(self.reward_predictor.parameters(), lr=1e-3)
        self.criterion = nn.BCEWithLogitsLoss()
        self.use_learned_reward = False  
        self.collecting_segments = True  # Active la collecte des segments par défaut


    def step(self, action):
        """Exécute une action et collecte les données pour l'entraînement des préférences."""
        step_result = self.env.step(action)  # Utiliser self.env au lieu de env

        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:  # Cas où seulement 4 valeurs sont retournées (ancienne version de Gym)
            obs, reward, done, info = step_result
            terminated, truncated = done, False  # Assigne truncated à False par défaut

        done = terminated or truncated  

        if self.collecting_segments:
            self._update_episode_segment(obs, reward, done)

        if self.use_learned_reward:
            predicted_reward = self.reward_predictor.forward(torch.tensor(obs, dtype=torch.float32)).item()
            return obs, predicted_reward, terminated, truncated, info
        else:
            return obs, reward, terminated, truncated, info


    def switch_to_predicted_reward(self):
        self.use_learned_reward = True

    def _update_episode_segment(self, obs, reward, done):
        """Ajoute une observation et enregistre le segment si nécessaire."""
        self.current_segment.append((obs, reward))

        if done or len(self.current_segment) >= self.segment_length:
            self._store_segment()
            self.current_segment = []

            if len(self.segments) >= 2:
                self._compare_segments()
    def _store_segment(self):
        """Ajoute un segment à la mémoire."""
        segment = list(self.current_segment)
        self.segments.append(segment)
        self.pref_interface.add_segment(segment)

    def _compare_segments(self):
        """Affiche la comparaison et entraîne le modèle en fonction de la préférence utilisateur."""
        pref = self.pref_interface.query_user()

        if pref:
            print("🎯 Préférence reçue, entraînement en cours...")
            s1, s2, preference = pref
            loss = self.reward_predictor.train_model(s1, s2, preference, self.optimizer, self.criterion)
            print(f"✅ Loss: {loss}")

