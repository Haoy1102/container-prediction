import pickle
import gym
import numpy as np
from gym import spaces
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import re
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContainerCacheEnv(gym.Env):
    def __init__(self, data, parameters, queue_length=5):
        max_remain_slices, max_cache_num, cache_slice_num, alpha, type_num = parameters
        # 初始化环境
        self.data = data  # 数据集
        self.time_step = 0  # 当前时间片
        self.max_remain_slices = max_remain_slices  # 最大缓存时间片数
        self.max_cache_num = max_cache_num  # 最大缓存容器数
        self.cache_slice_num = cache_slice_num
        self.alpha = alpha  # 缓存开销系数
        self.type_num = type_num
        self.cached_containers = deque(maxlen=queue_length)  # 缓存的容器
        self.total_cost = 0  # 总开销
        self.total_start_cost = 0

    def step(self, action):
        # 执行动作
        done = False
        # 清除到期容器
        self.clear_expired_containers()
        # 计算开销
        self.total_cost += self.calculate_cost()
        # 针对action更新状态空间
        self.current_state = self.update_state_space(action)
        # 更新时间片
        self.time_step += 1
        if self.time_step == len(self.data):
            done = True
        else:
            # 没结束，将下一时刻的data组合放入state，以供模型学习
            self.compose_state()
        return self.current_state, {}, done, {}

    def reset(self):
        # 重置环境
        self.time_step = 0
        self.current_state = np.zeros(self.type_num * 2).astype(int)
        np_slice = np.unique(np.array(self.data[self.time_step]))
        # 将np_slice中的所有值加上6
        np_slice += self.type_num
        indexs = np_slice.tolist()
        self.current_state[indexs] = 1
        self.cached_containers = deque(maxlen=self.cached_containers.maxlen)
        self.total_cost = 0
        self.total_start_cost = 0
        return self.current_state

    def update_state_space(self, action):
        # 更新状态空间
        for i in range(len(action)):
            if action[i] > 0 and i not in self.cached_containers:
                # 启动容器
                self.cached_containers.append(i)
        # 对已缓存容器更新缓存时间
        for i in self.cached_containers:
            if action[i] > 0:
                self.current_state[i] = self.cache_slice_num
        return self.current_state

    def calculate_cost(self):
        # 计算开销
        np_slice = np.array(self.data[self.time_step])
        # 找出np_slice中所有数值非0的项,即请求
        # nonzero_values = np.extract(np_slice != 0, np_slice)
        # 计算不在self.current_state中的非0项数量
        startup_cost = 0
        for value in np_slice:
            if self.current_state[value] == 0:
                # 启动容器开销
                startup_cost += 1
        self.total_start_cost += startup_cost
        # 缓存容器开销
        cache_cost = len(np.extract(self.current_state[:self.type_num] != 0, self.current_state[:self.type_num]))
        # 总开销
        cost = startup_cost + self.alpha * cache_cost
        return cost

    def clear_expired_containers(self):
        # 清除到期容器
        self.current_state[:self.type_num][self.current_state[:self.type_num] > 0] -= 1
        self.cached_containers = deque([i for i in self.cached_containers if self.current_state[i] > 0], maxlen=self.cached_containers.maxlen)

    def compose_state(self):
        # 将下一时刻的请求放入state
        self.current_state[self.type_num:] = 0
        np_slice = np.unique(np.array(self.data[self.time_step]))
        # 将np_slice中的所有值加上15
        np_slice += self.type_num
        indexs = np_slice.tolist()
        self.current_state[indexs] = 1


class FixCache:
    def __init__(self, env):
        self.env = env


    def act(self, state):
        time_step = self.env.time_step
        np_slice = np.array(self.env.data[time_step])

        # 如果当前时间片没有任何请求，则返回全为0的动作向量
        if np_slice.size == 0:
            return np.zeros(self.env.type_num).astype(int)
        # 如果请求了k和m容器，则只在k和m位置上做预测，预测n个时间片
        else:
            # 固定缓存即可
            np_slice_unique = np.unique(np_slice)
            indices = np_slice_unique.tolist()

            full_predicted_actions = np.zeros(self.env.type_num).astype(int)
            full_predicted_actions[indices] = 1
            return full_predicted_actions
            # 将预测结果插入到全局动作向量中

    def test(self):
        state = self.env.reset()
        done = False
        while not done:
            action=self.act(state)
            state, _, done, _ = self.env.step(action)
            cache_cost = self.env.total_cost - self.env.total_start_cost
        return self.env.total_cost, self.env.total_start_cost, cache_cost


data = [[0, 4, 4],
        [],
        [4],
        [3]]

env = ContainerCacheEnv(data, [4, 4, 4, 1, 10], queue_length=5)
fix_cache = FixCache(env)
print(fix_cache.test())
