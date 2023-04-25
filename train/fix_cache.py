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

class ContainerCacheEnv(gym.Env):
    def __init__(self, data, parameters):
        max_remain_slices, max_cache_num, cache_slice_num, alpha, type_num = parameters
        # 初始化环境
        self.data = data  # 数据集
        self.time_step = 0  # 当前时间片
        self.max_remain_slices = max_remain_slices  # 最大缓存时间片数
        self.max_cache_num = max_cache_num  # 最大缓存容器数
        self.cache_slice_num = cache_slice_num
        self.alpha = alpha  # 缓存开销系数
        self.type_num = type_num
        self.cached_containers = []  # 缓存的容器
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
        self.cached_containers = []
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
        self.cached_containers = np.where(self.current_state[:self.type_num] > 0)[0].tolist()

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


# data = [[0, 4, 4],
#         [],
#         [4],
#         [3]]
# ----------------------------------------------------------------
def transform_data(request):
    # 提取特征
    uri = request["uri"]
    timestamp = request["timestamp"]
    return [uri, timestamp]

def preprocess_data(raw_data,interval):
    # 将原始数据转换为数据帧
    df = pd.DataFrame(raw_data, columns=["container_type", "timestamp"])

    # 计算每个容器类型出现的频率
    freq = df["container_type"].value_counts()

    # 将出现频率低于阈值的类型替换为"else"
    low_freq_types = freq[freq <= 270].index
    df["container_type"].replace(low_freq_types, "miscellaneous", inplace=True)
    # low_freq_types = freq[(freq > 2000) & (freq <= 4000)].index
    # df["container_type"].replace(low_freq_types, "small", inplace=True)

    # 使用标签编码
    container_type = df["container_type"].values
    label_encoder = LabelEncoder()
    container_type = label_encoder.fit_transform(container_type)

    # 将ISO 8601格式的时间戳转换为pandas的时间戳，并将时区设置为UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # 按照每n秒一个时间片进行分割
    start_time = pd.Timestamp("2017-06-21T06:00:00.000Z", tz='UTC')
    end_time = pd.Timestamp("2017-06-21T12:00:00.000Z", tz='UTC')
    time_slices = pd.date_range(start=start_time, end=end_time, freq=interval)
    container_type_slices = []
    for i in range(len(time_slices) - 1):
        start_time_slice = time_slices[i]
        end_time_slice = time_slices[i + 1]
        container_type_slice = container_type[
            (df["timestamp"] >= start_time_slice) & (df["timestamp"] < end_time_slice)]
        container_type_slices.append(container_type_slice)

    return container_type_slices

def load_data_generator(json_files):
    for json_file in tqdm(json_files, desc="Loading JSON files"):
        with open(json_file, 'r') as f:
            try:
                requests = json.load(f)
                for request in requests:
                    data = transform_data(request)
                    yield data
            except json.JSONDecodeError as json_err:
                print("JSONDecodeError: " + str(json_err) + " in JSON file: " + json_file)
            except ValueError as value_err:
                print("ValueError: " + str(value_err) + " in JSON file: " + json_file)

def load_json_files(data_path):
    json_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".json"):
                json_file = os.path.join(root, file)
                json_files.append(json_file)
    return json_files

data_path = "../dataset/node-dal09-78-6.21"
# n秒为1个时间片 10s相当于将数据量扩大10倍
interval="1S"

json_files = load_json_files(data_path)
raw_data = load_data_generator(json_files)
data = preprocess_data(raw_data,interval)

print("数据处理完成")

max_cache_num = 10  # 最大缓存容器数-无用
alpha = 0.0001  # 缓存开销系数
cache_slice_num = 1/alpha
type_num = 15
max_remain_slices = 1000  # 最大缓存时间片数-无用
parameters_env = max_remain_slices, max_cache_num, cache_slice_num, alpha, type_num

# 初始化环境和DQNAgent
env = ContainerCacheEnv(data, parameters_env)
agent = FixCache(env)

# 测试模型
cost = agent.test()
total_cost, start_cost, cache_cost = cost
print("------------test-------------")
print("Total cost:", total_cost)
print("Start cost:", start_cost)
print("Cache cost:", cache_cost)
