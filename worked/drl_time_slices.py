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
import datetime
import random
import gym
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import gym
import numpy as np

class EdgeNodeEnv(gym.Env):
    def __init__(self, data, memory_size,time_slice_len):
        self.data = data  # 数据集合
        self.memory_size = memory_size  # 内存大小

        # 定义动作空间和观察空间。
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-1, high=1000, shape=(memory_size + 2,), dtype=float)

        self.containers_in_memory = []  # 当前在内存中的容器集合

        # 定义时间片长度
        self.time_slice_len = time_slice_len

        # 定义内存中缓存的容器列表
        self.cache_list = []
        # 定义时间片计数器
        self.time_step = 0


    def reset(self):

        self.state = np.zeros(n)
        self.request_list = []
        self.cache_list = []
        self.time_step = 0

        self.containers_in_memory.clear()
        # 初始化 current_step 变量为第一个请求位置.
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        hit = False
        done = False
        # 获取当前请求容器类型
        container_type = self.data[self.current_step]
        if container_type in self.containers_in_memory:
            reward = 3.0  # 命中缓存，给予正向奖励
            hit = True
        else:
            reward = -1.0  # 没有命中缓存，给予负向奖励
            if len(self.containers_in_memory) >= self.memory_size:
                # 内存已满, 需要清除一个容器再加载新容器
                containers_to_remove = []
                for c in self.containers_in_memory:
                    if np.random.uniform() < (1 / len(self.containers_in_memory)):
                        containers_to_remove.append(c)

                if not containers_to_remove:  # 如果没有找到要删除的容器，则随机选择一个进行删除
                    container_to_remove = np.random.choice(list(self.containers_in_memory))
                    self.containers_in_memory.remove(container_to_remove)

                else:  # 找到了要删除的容器，则从列表中移除即可
                    for c in containers_to_remove:
                        self.containers_in_memory.remove(c)

                self.containers_in_memory.add(container_type)

            else:
                # 内存未满, 直接将新容器加入内存
                self.containers_in_memory.add(container_type)

            if action == 0:  # 不保留缓存
                if container_type in self.containers_in_memory:
                    self.containers_in_memory.remove(container_type)

            elif action == 1:  # 保留缓存
                pass

        if (self.current_step + 1) == len(self.data):
            done = True

        info = {'hit': hit}
        observation = self._get_observation()
        self.current_step += 1  # 更新当前处理的请求位置.

        return observation, reward, done, info

    def _get_observation(self):
        """
        返回当前环境观测到的状态信息.
        当前请求的容器类型是否在内存中。如果是，则表示已经命中缓存；如果不是，则需要加载到内存。
        内存利用率。可以通过计算当前内存使用情况与总内存大小之比来得出。
        缓存在内存中的其他容器信息。例如，在清除一个缓存在内存中的容器时，可能希望选择最少被访问过或者大小较小等策略进行清理。
        Returns :
            ndarray : 状态信息数组.
        """
        # 获取当前请求容器类型
        container_type = self.data[self.current_step]

        # 计算当前内存利用率
        memory_utilization = len(self.containers_in_memory) / self.memory_size

        # 创建一个长度为 self.memory_size 的全零数组
        observation = np.full(shape=(self.memory_size,), fill_value=-1)
        containers_sorted = sorted(self.containers_in_memory)
        # 将缓存在内存中的其他容器信息填充到 observation 中
        for i in range(len(containers_sorted)):
            observation[i] = containers_sorted[i]

        # 在 observation 数组开头添加当前请求容器类型是否在内存中和当前内存利用率两个元素
        observation = np.concatenate(([int(container_type in self.containers_in_memory), memory_utilization],
                                      observation))

        return observation

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
        self.model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
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

def preprocess_data(raw_data, threshold=270):
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

memory_size = 10
time_slice_len = 5400
len_data = len(data)
rewards = []
hit_rates = []

env = EdgeNodeEnv(data, memory_size,time_slice_len)
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
