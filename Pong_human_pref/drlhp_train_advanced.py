import numpy as np
import collections
import pickle
import zlib
import copy
import gzip
import multiprocessing as mp
import time
import random


class PrefDB:
    """
    Une base de données circulaire pour stocker les préférences entre segments.
    """

    def __init__(self, maxlen=100):
        self.segments = {}  # Stocke les segments sans duplication
        self.seg_refs = {}  # Nombre de références à chaque segment
        self.prefs = []  # Liste circulaire des préférences
        self.maxlen = maxlen  # Taille maximale de la base

    def append(self, s1, s2, pref):
        """Ajoute une nouvelle préférence en évitant les doublons"""
        k1 = hash(np.array(s1).tobytes())
        k2 = hash(np.array(s2).tobytes())

        for k, s in zip([k1, k2], [s1, s2]):
            if k not in self.segments:
                self.segments[k] = s
                self.seg_refs[k] = 1
            else:
                self.seg_refs[k] += 1

        self.prefs.append((k1, k2, pref))

        if len(self.prefs) > self.maxlen:
            self.del_first()

    def del_first(self):
        """Supprime la plus ancienne préférence"""
        self.del_pref(0)

    def del_pref(self, n):
        """Supprime une préférence spécifique"""
        k1, k2, _ = self.prefs[n]
        for k in [k1, k2]:
            if self.seg_refs[k] == 1:
                del self.segments[k]
                del self.seg_refs[k]
            else:
                self.seg_refs[k] -= 1
        del self.prefs[n]

    def __len__(self):
        return len(self.prefs)

    def save(self, path):
        """Sauvegarde la base de préférences"""
        with gzip.open(path, 'wb') as pkl_file:
            pickle.dump(copy.deepcopy(self), pkl_file)

    @staticmethod
    def load(path):
        """Charge une base de préférences depuis un fichier"""
        with gzip.open(path, 'rb') as pkl_file:
            return pickle.load(pkl_file)


class PrefInterface:
    """Interface utilisateur demandant des préférences en parallèle."""

    def __init__(self, max_segments=100):
        self.max_segments = max_segments
        self.tested_pairs = set()

    def run(self, seg_queue, pref_queue, kill_flag, parent_conn):
        """Boucle principale pour collecter des préférences en parallèle."""
        segments = []
        while not kill_flag.value:
            while not seg_queue.empty():
                segment = seg_queue.get()
                if len(segments) < self.max_segments:
                    segments.append(segment)
                else:
                    segments.pop(0)
                    segments.append(segment)

            if len(segments) < 2:
                time.sleep(1)
                continue

            s1, s2 = random.sample(segments, 2)
            if (s1, s2) in self.tested_pairs:
                continue

            self.tested_pairs.add((s1, s2))

            # Envoyer la question au processus principal
            parent_conn.send((s1, s2))

            # Attendre la réponse du processus principal
            choice = parent_conn.recv()

            if choice == "1":
                pref_queue.put((s1, s2, (1.0, 0.0)))
            elif choice == "2":
                pref_queue.put((s1, s2, (0.0, 1.0)))

        print("🔴 Arrêt du processus de collecte des préférences")


class RewardTrainer:
    """Entraîne le modèle de récompense en parallèle."""

    def __init__(self):
        self.model = {}  # Remplace par ton modèle réel

    def train(self, pref_queue, kill_flag):
        """Boucle principale d'entraînement"""
        while not kill_flag.value:
            if not pref_queue.empty():
                s1, s2, pref = pref_queue.get()
                print(f"📊 Préférence reçue : {pref}")

                # Simuler un entraînement
                time.sleep(1)
                print("✅ Entraînement terminé pour cette préférence")

        print("🔴 Arrêt du processus d'entraînement")


def stop_processes(kill_flag, pref_process, train_process):
    """Arrête proprement les processus."""
    print("🛑 Arrêt des processus en cours...")
    kill_flag.value = 1
    pref_process.join()
    train_process.join()
    print("✅ Processus arrêtés proprement.")


def start_processes():
    """Crée les processus et les files de communication."""
    seg_queue = mp.Queue()
    pref_queue = mp.Queue()
    kill_flag = mp.Value('i', 0)

    # Pipe de communication entre le processus principal et PrefInterface
    parent_conn, child_conn = mp.Pipe()

    # Interface de préférences
    pref_interface = PrefInterface()
    pref_process = mp.Process(target=pref_interface.run, args=(seg_queue, pref_queue, kill_flag, child_conn))

    # Lancer le processus
    pref_process.start()
    print(f"🚀 Processus de l'interface de préférence démarré (PID={pref_process.pid})")

    return seg_queue, pref_queue, kill_flag, pref_process, parent_conn


if __name__ == "__main__":
    try:
        seg_queue, pref_queue, kill_flag, pref_process, parent_conn = start_processes()

        # Lancer l'entraînement du modèle dans un second processus
        trainer = RewardTrainer()
        train_process = mp.Process(target=trainer.train, args=(pref_queue, kill_flag))
        train_process.start()

        # Simuler l'ajout de segments
        for i in range(5):  # Ajoute un nombre limité de segments
            seg_queue.put(f"Segment {i}")
            time.sleep(0.5)

        # Gérer l'interface utilisateur dans le **processus principal**
        while True:
            if parent_conn.poll():  # Vérifier si une question est posée
                s1, s2 = parent_conn.recv()
                print("🧐 Comparez ces segments :")
                print(f"1: {s1}")
                print(f"2: {s2}")
                choice = input("Quel segment préférez-vous ? (1/2) ou (Enter pour passer) : ").strip()
                parent_conn.send(choice)  # Envoyer la réponse au sous-processus

    except KeyboardInterrupt:
        print("\n🛑 Interruption détectée ! Arrêt propre des processus...")
        stop_processes(kill_flag, pref_process, train_process)

    finally:
        stop_processes(kill_flag, pref_process, train_process)
