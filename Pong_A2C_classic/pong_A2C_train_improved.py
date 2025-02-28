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
        state = state / 255.0  # Normalisation de l'état
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
        state = state / 255.0  # Normalisation de l'état
        return self.network(state)


# Generalized Advantage Estimation (GAE)
def compute_gae(rewards, values, gamma=GAMMA, lambda_=LAMBDA_GAE):
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32)  # Convertir en tensor PyTorch
    values = [torch.tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v for v in values]
    values = torch.stack(values, dim=0).detach()  # Stacker avec dim=0

    gae = torch.zeros_like(values[0])  # Initialiser un tenseur de même taille que values[0]
    returns = []

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma * lambda_ * gae
        returns.insert(0, gae + values[i])

    return torch.stack(returns, dim=0)


# Boucle d'entraînement Advantage Actor-Critic
def advantage_actor_critic(env, max_episodes, learning_rate, gamma):
    actor = Actor((N_STACK, 84, 84), env.action_space.n)
    critic = Critic((N_STACK, 84, 84))
    actor_optimizer = optim.RMSprop(actor.parameters(), lr=learning_rate, alpha=0.99, eps=1e-5)
    critic_optimizer = optim.RMSprop(critic.parameters(), lr=learning_rate, alpha=0.99, eps=1e-5)

    # Schedulers pour Learning Rate
    scheduler_actor = torch.optim.lr_scheduler.StepLR(actor_optimizer, step_size=50000, gamma=0.99)
    scheduler_critic = torch.optim.lr_scheduler.StepLR(critic_optimizer, step_size=50000, gamma=0.99)

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

            values.append(value)  # Ajouter V(s)
            log_probas.append(log_proba)
            rewards.append(reward)

            if done[0]:  # Vérification correcte avec VecEnv
                break

            state = next_state

        # **Correction : Ajouter V(s_T)**
        last_value = critic(next_state).squeeze().detach()
        values.append(last_value)  # Ajout de V(s_T) pour éviter l'IndexError

        # **Correction : `values` doit être `len(rewards) + 1`**
        returns = compute_gae(rewards, values, gamma, LAMBDA_GAE)

        # Calcul de l'avantage
        values = torch.stack(values[:-1])  # Retirer V(s_T) pour le calcul de l'avantage
        advantage = returns - values

        # **Normalisation de l'advantage avant l'entraînement**
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Reshape log_probas pour qu'il corresponde à la taille de advantage
        log_probas = torch.cat(log_probas, dim=0).view(len(rewards), env.num_envs)

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

        # Mise à jour des schedulers de Learning Rate
        scheduler_actor.step()
        scheduler_critic.step()

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

