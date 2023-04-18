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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
import gym
import numpy as np
from gym import spaces


class ContainerCacheEnv(gym.Env):
    def __init__(self, data, memory_size, alpha,max_remain_slices):
        self.data = data
        self.alpha = alpha
        self.memory_size = memory_size
        self.max_remain_slices=max_remain_slices
        self.action_space = spaces.MultiDiscrete([max_remain_slices+1]*15)
        self.observation_space = spaces.MultiDiscrete([max_remain_slices+1]*15)
        self.reset()

    def reset(self):
        self.total_cost = 0
        self.cur_time = 0
        self.observation_space = np.zeros(15, dtype=int)
        return self.observation_space

    def step(self, action):
        container_type, cache_time = action
        cold_start_cost = 0

        # 检查容器是否已经在内存中
        if self.memory[container_type] > 0:
            self.remaining_time[container_type] = cache_time
        else:
            cold_start_cost += 1
            if np.sum(self.memory) == memory_size:
                # 剔除剩余时间最少的容器
                evict_idx = np.argmin(self.remaining_time)
                self.memory[evict_idx] = 0
                self.remaining_time[evict_idx] = 0
            self.memory[container_type] = 1
            self.remaining_time[container_type] = cache_time

        # 更新缓存中的容器的剩余时间
        self.remaining_time[self.memory > 0] -= 1

        # 计算缓存代价
        cache_cost = np.sum(self.remaining_time[self.memory > 0])
        # 计算总代价
        total_cost = cold_start_cost + self.alpha * cache_cost
        # 更新总代价
        self.total_cost += total_cost
        # 更新时间步数
        self.cur_time += 1

        # 检查是否结尾
        if self.cur_time >= len(self.data):
            done = True
        else:
            done = False
            self.cur_request = self.data[self.cur_time]

        # 返回 new state, reward, done
        return (self.memory, self.remaining_time), -total_cost, done, {}

    def get_dqn_params(self):
        input_size = self.observation_space.shape[0]
        output_size = self.action_space.shape[0]
        return input_size, output_size

    def render(self, mode='human'):
        pass



class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, env,parameters):

        # 初始化环境和超参数
        gamma, epsilon_start, epsilon_end,\
        epsilon_decay, learning_rate, memory_capacity = parameters
        self.env = env
        self.gamma = gamma  # 折扣因子
        self.epsilon_start = epsilon_start  # 贪婪策略起始值
        self.epsilon_end = epsilon_end  # 贪婪策略结束值
        self.epsilon_decay = epsilon_decay  # epsilon线性衰减的步数
        self.learning_rate = learning_rate

        # 初始化经验回放缓冲区、当前状态和epsilon值。

        self.memory_capacity = memory_capacity
        self.memory = []

        self.current_state = None
        self.current_epsilon = self.epsilon_start

        # 将模型和优化器放置在GPU上
        input_size,output_size = env.get_dqn_params()
        self.model = DQN(input_size, output_size).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        # 根据当前状态选择动作。

        device = next(self.model.parameters()).device  # 获取设备信息

        # 随机选择动作
        if random.random() < self.current_epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.model(torch.tensor(state).float().to(device))
                return q_values.argmax().item()

    def replay_memory(self, batch_size):
        # 从经验回放缓冲区中随机抽取一批样本，并使用它们更新模型参数。

        device = next(self.model.parameters()).device  # 获取设备信息

        if len(self.memory) < batch_size:
            return

        transitions = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states = zip(*transitions)

        states = np.array(states)
        next_states = np.array(next_states)
        states = torch.tensor(states).float().to(device)
        actions = torch.tensor(actions).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards).unsqueeze(1).to(device)
        next_states = torch.tensor(next_states).float().to(device)

        with torch.no_grad():
            target_q_values = self.gamma * self.model(next_states).max(dim=-1)[0].unsqueeze(-1) + rewards
        current_q_values = self.model(states).gather(1, actions)
        loss = self.loss_fn(current_q_values, target_q_values)

        # 更新模型参数。
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_step(self):
        # 获取当前观察值并选择动作。
        state = self.current_state
        action = self.act(state)

        next_state, reward, done, info = self.env.step(action)
        # 将经验存储到经验回放缓冲区中。
        self.memory.append((state, action, reward, next_state))

        # 从经验回放缓冲区中随机抽取一批样本，并使用它们更新模型参数。
        self.replay_memory(batch_size=32)

        # 更新当前状态和epsilon值，然后返回奖励信息。
        self.current_state = next_state
        self.current_epsilon = max(self.epsilon_end,
                                   self.current_epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay)

        return reward, done, info

    def train(self, num_epochs, len_data):
        total_reward = 0.0
        total_hit = 0.0
        self.current_state = env.reset()
        for i in range(num_epochs):
            reward, done, info = self.train_step()
            total_reward += reward
            total_hit += 1 if info.get('hit') else 0
            if done:
                print("done, used data num:{}".format(i))
                break
            # print("num_epochs:{}".format(i))
        hit_rate = total_hit / len_data
        print("hit_rate:{}".format(hit_rate))
        print("total_reward:{}".format(total_reward))
        return total_reward, hit_rate

# ----------------------------------------------------------------
def transform_data(request):
    # 提取特征
    uri = request["uri"]
    timestamp = request["timestamp"]

    # match = re.search(r"v2/([^/]+)/([^/]+)", uri)
    # if match:
    #     uri_parts = match.groups()
    #     # uri = "/".join(uri_parts)
    #     uri = uri_parts[0]
    # else:
    #     uri = "v2"

    return [uri,timestamp]

def preprocess_data(raw_data, threshold=280):
    # 将原始数据转换为数据帧
    df = pd.DataFrame(raw_data, columns=["container_type","timestamp"])

    # 计算每个容器类型出现的频率
    freq = df["container_type"].value_counts()

    # 将出现频率低于阈值的类型替换为"else"
    low_freq_types = freq[freq < threshold].index
    df["container_type"].replace(low_freq_types, "else", inplace=True)

    # 使用标签编码
    container_type = df["container_type"].values
    label_encoder = LabelEncoder()
    container_type = label_encoder.fit_transform(container_type)

    # 将ISO 8601格式的时间戳转换为pandas的时间戳，并将时区设置为UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # 按照每4秒一个时间片进行分割
    start_time = pd.Timestamp("2017-06-21T06:00:00.000Z", tz='UTC')
    end_time = pd.Timestamp("2017-06-21T12:00:00.000Z", tz='UTC')
    time_slices = pd.date_range(start=start_time, end=end_time, freq="1S")
    container_type_slices = []
    for i in range(len(time_slices) - 1):
        start_time_slice = time_slices[i]
        end_time_slice = time_slices[i+1]
        container_type_slice = container_type[(df["timestamp"] >= start_time_slice) & (df["timestamp"] < end_time_slice)]
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

# 加载数据集

def load_json_files(data_path):
    json_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".json"):
                json_file = os.path.join(root, file)
                json_files.append(json_file)
    return json_files


data_path = "../dataset/node-dal09-78-6.21"

json_files = load_json_files(data_path)
raw_data = load_data_generator(json_files)
data = preprocess_data(raw_data)

gamma = 0.99  # 折扣因子
epsilon_start = 1.0  # 贪婪策略起始值
epsilon_end = 0.01  # 贪婪策略结束值
epsilon_decay = 5000  # epsilon线性衰减的步数
learning_rate = 0.001
# 初始化经验回放缓冲区、当前状态和epsilon值。
memory_capacity = 10000
parameters = gamma, epsilon_start, epsilon_end, \
             epsilon_decay, learning_rate, memory_capacity

max_remain_slices = 1000
alpha=0.004
memory_size = 10
len_data = len(data)
rewards = []
hit_rates = []

# data = [[1, 0, 0, 0],
#         [0, 0, 0, 0],
#         [4, 0, 0, 0],
#         [12, 0, 0, 0]]

env = ContainerCacheEnv(data, memory_size,alpha,max_remain_slices)
agent = DQNAgent(env,parameters)
for i in range(100):
    result = agent.train(num_epochs=5000, len_data=len_data)
    reward, hit_rate = result
    rewards.append(reward)
    hit_rates.append(hit_rate)
    print("epoch:{}".format(i))
    print("-----------------")

# plt.switch_backend('TkAgg')
plt.plot(range(len(rewards)), rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

plt.plot(range(len(hit_rates)), hit_rates)
plt.xlabel("Episode")
plt.ylabel("Hit rates")
plt.show()
