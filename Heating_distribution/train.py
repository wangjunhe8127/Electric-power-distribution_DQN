import gym
import numpy as np
import torch as th
import GymEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import random
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from PARAMS_SN import TARGET
from PARAMS import DQN_PARAMS
import os


class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box,features_dim:int):
        super(CustomMLP, self).__init__(observation_space, features_dim)
        self.net = th.nn.Sequential(
                             th.nn.Linear(3, 17),
                             th.nn.LeakyReLU(),
                             th.nn.Linear(17,35),
                             th.nn.PReLU())
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)
def fixed_seed(i):
    random.seed(i)
    os.environ['PYTHONHASHSEED'] = str(i)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(i)
    th.manual_seed(i)
    th.cuda.manual_seed(i)
    th.cuda.manual_seed_all(i)  # if you are using multi-GPU.
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
if __name__=='__main__':
    fixed_seed(1)#固定随机种子
    params = DQN_PARAMS()
    env = make_vec_env("Heating_distribution-v0", n_envs=10,seed = 1)
    model = DQN(
        "MlpPolicy", 
        env=env, 
        learning_rate=5e-4,
        gamma=0.99,
        batch_size=500,
        buffer_size=20000,
        learning_starts=0,
        target_update_interval=250,
        policy_kwargs={"net_arch" : [256,256,128,128]},
        verbose=0,
        tensorboard_log="../tf-logs/")
    
    model.learn(
        total_timesteps=2*10**5,
        n_eval_episodes=10,)
    model.save('%s.pkl'%repr(TARGET))