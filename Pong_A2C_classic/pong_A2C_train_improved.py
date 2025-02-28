import cv2
import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
GAMMA = 0.99
LR = 1e-4
ENTROPY_BETA = 0.01
HIDDEN_SIZE = 512
LAMBDA_GAE = 0.95
N_ENVS = 8  # Nombre d'environnements parallèles
N_STACK = 4  # Nombre de frames empilées
MAX_EPISODES = 1000
USE_PARALLEL_ENV = True  # Si False, utilise DummyVecEnv à la place de SubprocVecEnv

# TensorBoard Logging
log_dir = "./tensorboard_logs/a2c_pong/"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)


def make_env():
    def _env():
        env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        env = AtariWrapper(env)
        return env
    return _env


# Modèles Actor et Critic
class Actor(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, n_actions),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, state):
        return self.network(state)


# Generalized Advantage Estimation (GAE)
def compute_gae(rewards, values, gamma=GAMMA, lambda_=LAMBDA_GAE):
    rewards = torch.tensor(rewards, dtype=torch.float32)  # Convertir les rewards en tenseurs
    values = torch.tensor(values, dtype=torch.float32)  # Convertir les valeurs en tenseurs
    
    gae = 0
    returns = []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1].squeeze() - values[i].squeeze()
        gae = delta + gamma * lambda_ * gae
        returns.insert(0, gae + values[i].squeeze())

    return torch.tensor(returns, dtype=torch.float32)


# Boucle d'entraînement Advantage Actor-Critic
def advantage_actor_critic(env, max_episodes, learning_rate, gamma):
    actor = Actor((N_STACK, 84, 84), env.action_space.n)
    critic = Critic((N_STACK, 84, 84))
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

    for episode in range(max_episodes):
        log_probas, values, rewards = [], [], []
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        for step in range(1000):  # Limite du nombre de steps par épisode
            value = critic(state).squeeze().detach()
            policy = actor(state)
            
            action = torch.multinomial(policy, num_samples=1).squeeze().cpu().numpy()
            log_proba = torch.log(policy.gather(1, torch.tensor(action).unsqueeze(1)))
            
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            values.append(value)
            log_probas.append(log_proba)
            rewards.append(reward)

            if done[0]:  # Vérification correcte avec VecEnv
                break

            state = next_state

        # Calcul des Q-values avec GAE
        values.append(critic(next_state).squeeze().detach())
        returns = compute_gae(rewards, values, gamma, LAMBDA_GAE)
        values = torch.tensor(values[:-1], dtype=torch.float32)

        # Calcul de l'avantage
        advantage = returns - values
        log_probas = torch.cat(log_probas)

        # Calcul des pertes Actor-Critic
        actor_loss = (-log_probas * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        entropy = -(policy * torch.log(policy + 1e-10)).sum(dim=1).mean()
        total_loss = actor_loss + critic_loss - ENTROPY_BETA * entropy

        # Mise à jour des modèles avec gradient clipping
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
        actor_optimizer.step()
        critic_optimizer.step()

        # Log reward dans TensorBoard
        writer.add_scalar("rollout/total_reward", np.sum(rewards), episode)
        print(f"Episode {episode}: Reward = {np.sum(rewards)}")

    return


# Exécution principale protégée pour Windows
if __name__ == "__main__":
    # Choix entre SubprocVecEnv (multi-processus) et DummyVecEnv (single-process)
    if USE_PARALLEL_ENV:
        env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])  # 8 environnements en parallèle
    else:
        env = DummyVecEnv([make_env])  # 1 seul environnement, sans multiprocessing
    
    env = VecFrameStack(env, n_stack=N_STACK, channels_order='last')
    env = VecTransposeImage(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Lancement de l'entraînement
    advantage_actor_critic(env, MAX_EPISODES, learning_rate=LR, gamma=GAMMA)

    # Fermeture du logger
    writer.close()
