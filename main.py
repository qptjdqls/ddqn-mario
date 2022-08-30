'''
!pip install gym-super-mario-bros===7.3.0
'''
from pathlib import Path
from collections import deque
import random
import datetime
import os
import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace # NES Emulator for OpenAI Gym
import gym_super_mario_bros             # Super Mario environment for OpenAI Gym



'''Initialize Environment'''
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
# Limit the action-space to 0. walk right 1. jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])
env.reset()
next_state, reward, done, info = env.step(action=0)
print(f"{next_state.shape}, \n {reward}, \n {done}, \n {info}")

'''Preprocess Environemnt'''
class SkipFrame(gym.Wrapper):
    '''
    `SkipFrame` is a custom wrapper that inherits from `gym.Wrapper` and implements
    the `step()` function. Because consecutive frames don't vary much, we can skip
    n-intermediate frames wihtout losing much information. The n-th frame aggregates
    rewards accumulated over each skipped frame.
    '''
    def __init__(self, env, skip):
        '''Return only every `skip`-th frame'''
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        '''Repeat action, and sum reward'''
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

