import gym
import math
import numpy as np
from .utility1 import ABC, people_num, delta_t
from .utility2 import resolution
from .utility3 import Reward
gym.logger.set_level(40)
class Env(gym.Env):
    # 初始化参数
    def __init__(self):
        self.abc = ABC()
        self.Reward = Reward()
        self.observation_space = gym.spaces.Box(low=-20, high=20, shape=(people_num,1))
        self.action_space = gym.spaces.Discrete(32)
        self.t = 0.0
    # 奖励函数
    def get_reward(self):
        rewards = self.Reward.com_rewards(self.abc.inside_now_temperature, self.abc.H_O_P, self.abc.price)
        self.past_H_O_P = self.abc.H_O_P
        return rewards
    def get_state(self):
        state = (self.abc.inside_now_temperature - 25.5)/3
        return  state
    # 主程序
    def step(self, action):
        action = resolution(action)
#         print(action)
        self.t = self.t + delta_t
        self.abc.sim_step(self.t, action)
        rewards = self.get_reward()
        state = self.get_state()
        done = True if abs(self.t -24.0)<0.01 else False
        return state, rewards, done, {}
    # 重置环境
    def reset(self):
        self.t = 0.0
        self.abc.reset_abc()
        self.Reward.reset()
        state = self.get_state()
        return state
if __name__ == '__main__':
    env = Env()
    env.reset()
    for i in range(48):
        env.step(np.array([1,0,1,0,0]).reshape(people_num,1))