import cv2
import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from torch.utils.tensorboard import SummaryWriter
import os

# Hyperparameters
GAMMA = 0.99
LR = 1e-4
ENTROPY_BETA = 0.01
BATCH_SIZE = 5
HIDDEN_SIZE = 512

# Simulation parameters
MAX_EPOCHS = 100000
MAX_EPISODES = 1000 # On augmente à 100 épisodes pour mieux voir l'évolution
N_STACK = 4  # Nombre de frames empilées

# TensorBoard Logging
log_dir = "./tensorboard_logs/a2c_pong/"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)


def make_env():
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")  
    env = AtariWrapper(env)  
    return env

env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=N_STACK, channels_order='last')
env = VecTransposeImage(env)
env = VecNormalize(env, norm_obs=True, norm_reward=False) 

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
        state = state / 255.0
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
        state = state / 255.0
        return self.network(state)

# Boucle d'entraînement Advantage Actor-Critic
def advantage_actor_critic(env, max_epochs, max_episodes, learning_rate, gamma):
    actor = Actor((N_STACK, 84, 84), env.action_space.n)
    critic = Critic((N_STACK, 84, 84))
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

    all_rewards = []

    for episode in range(max_episodes):
        log_probas, values, rewards = [], [], []
        state = env.reset()
        # print(state.shape)
        # state = np.transpose(state, (0, 3, 1, 2))  # (batch, H, W, C) devient (batch, C, H, W)
        state = torch.tensor(state, dtype=torch.float32)

        episode_entropy = 0

        for epoch in range(max_epochs):
            value = critic(state).squeeze().detach().numpy()
            policy = actor(state)
            policy_np = policy.detach().numpy().squeeze()

            # Exploration
            tau = 1.0
            policy_scaled = policy_np ** (1 / tau)
            policy_scaled /= np.sum(policy_scaled)

            action = np.array([np.random.choice(env.action_space.n, p=policy_scaled)])
            log_proba = torch.log(policy.squeeze(0)[action])
            
            entropy = -torch.sum(policy * torch.log(policy + 1e-10), dim=1).mean()
            episode_entropy += entropy.item()

            next_state, reward, done, info = env.step(action)  
            # next_state = np.transpose(next_state, (0, 3, 1, 2))  
            next_state = torch.tensor(next_state, dtype=torch.float32) 

            values.append(value)
            log_probas.append(log_proba)
            rewards.append(reward)

            if done.item() or epoch == max_epochs - 1: 
                q_value = critic(next_state).squeeze().detach().numpy() if not done.item() else 0
                sum_rewards = np.sum(rewards)
                all_rewards.append(sum_rewards)

                # Log reward dans TensorBoard
                writer.add_scalar("rollout/total_reward", sum_rewards, episode)

                print(f"Episode {episode}: Reward = {sum_rewards}")
                break

            state = next_state 

        # Calcul des Q-values
        values = torch.tensor(np.array(values), dtype=torch.float32)
        q_values = []
        q_value = critic(next_state).squeeze().detach().numpy() if not done.item() else 0  
        for reward in reversed(rewards):
            q_value = reward + gamma * q_value
            q_values.insert(0, q_value)

        q_values = torch.FloatTensor(q_values)
        log_probas = torch.stack(log_probas)

        # Calcul de l'avantage et des pertes Actor-Critic
        advantage = q_values - values
        actor_loss = (-log_probas * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        actor_critic_loss = actor_loss + critic_loss - ENTROPY_BETA * entropy.mean()

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        actor_critic_loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()

    return all_rewards


# Lancement de l'entraînement
all_rewards = advantage_actor_critic(env, MAX_EPOCHS, MAX_EPISODES, learning_rate=LR, gamma=GAMMA)

# Fermeture du logger
writer.close()


