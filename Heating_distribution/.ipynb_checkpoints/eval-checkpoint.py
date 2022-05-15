import gym
from stable_baselines3 import DQN
import os
import numpy as np
import torch as th
import random
from PARAMS_SN import TARGET
from train import fixed_seed
import GymEnv

score = 0
steps = 0
done = False
env = gym.make("Heating_distribution-v0")
model = DQN.load('%s.pkl'%repr(TARGET))# 导入模型
fixed_seed(1)#固定随机种子
for i in range(1):
    state = env.reset()
    done = False
    while not done:
        action,_ = model.predict(state,deterministic=True)
#         action = action.squeeze(0)
        state,reward,done,_ = env.step(action)
    