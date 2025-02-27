import gymnasium as gym
import os
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor

num_envs = 8  # ou 16, selon les ressources machine
log_dir = "./tensorboard_logs/a2c_pong_base_parallel"
os.makedirs(log_dir, exist_ok=True)

def make_env(seed, rank, log_dir):
    def _init():
        env = gym.make("PongNoFrameskip-v4")  
        env.seed(seed + rank)
        env = AtariWrapper(env)
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _init

# On crée plusieurs environnements
env_fns = [make_env(seed=0, rank=i, log_dir=log_dir) for i in range(num_envs)]
env = SubprocVecEnv(env_fns, start_method='fork')
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

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
    device="auto"
)

model.learn(total_timesteps=50_000_000)
model.save("a2c_pong_parallel")
print("Modèle A2C sauvegardé !")
