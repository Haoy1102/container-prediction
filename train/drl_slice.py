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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.observation_space = spaces.Box(low=0, high=self.max_remain_slices, shape=(self.type_num * 2,),
                                            dtype=np.int32)  # 状态空间
        self.action_space = spaces.MultiBinary(self.type_num)  # 动作空间
        self.current_state = np.zeros(self.type_num * 2).astype(int)  # 当前状态
        self.cached_containers = []  # 缓存的容器
        self.total_cost = 0  # 总开销
        self.total_start_cost = 0

    def step(self, action):
        # 执行动作
        done = False
        # 清除到期容器
        self.clear_expired_containers()
        # 计算开销
        step_cost, step_start_cost = self.calculate_cost()
        self.total_cost += step_cost
        self.total_start_cost += step_start_cost
        # 针对action更新状态空间
        self.current_state = self.update_state_space(action)
        # 更新时间片
        self.time_step += 1
        if self.time_step == len(self.data):
            done = True
        else:
            # 没结束，将下一时刻的data组合放入state，以供模型学习
            self.compose_state()
        reward = -step_cost
        return self.current_state, reward, done, {}

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
                self.current_state[i] = min(self.current_state[i] + action[i] * self.cache_slice_num,
                                            self.max_remain_slices)
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
        # 缓存容器开销
        cache_cost = len(np.extract(self.current_state[:self.type_num] != 0, self.current_state[:self.type_num]))
        # 总开销
        cost = startup_cost + self.alpha * cache_cost
        return cost, startup_cost

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


import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_shape)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


import random
from collections import deque
import torch.optim as optim


class DQNAgent:
    def __init__(self, env, parameters):
        gamma, epsilon, epsilon_min, epsilon_decay, batch_size = parameters
        self.env = env
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.model = DQN(env.observation_space.shape[0], 2 ** env.action_space.shape[0]).to(device)
        self.target_model = DQN(env.observation_space.shape[0], 2 ** env.action_space.shape[0]).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def act(self, state):
        time_step = self.env.time_step
        np_slice = np.array(self.env.data[time_step])

        # TODO 考虑改一下，当随机的时候，没有任何请求返回全0，不随机的时候。。。
        # 如果当前时间片没有任何请求，则返回全为0的动作向量
        if np_slice.size == 0:
            return np.zeros(self.env.action_space.shape[0]).astype(int)
        # 如果请求了k和m容器，则只在k和m位置上做预测，预测n个时间片
        else:
            # 在请求不为0的位置上做预测
            np_slice_unique = np.unique(np_slice)
            indices = np_slice_unique.tolist()
            # indices = np.where(self.env.data[time_step] > 0)[0]
            state_at_indices = state[indices]
            # 以epsilon的概率随机选择一个动作，以便探索新的策略
            if np.random.rand() <= self.epsilon:
                predicted_actions = self.env.action_space.sample()[indices].astype(int)
                full_predicted_actions = np.zeros(self.env.action_space.shape[0]).astype(int)
                full_predicted_actions[indices] = predicted_actions
                return full_predicted_actions
            # 否则，使用当前模型预测最优动作
            else:
                # 应该把self.env.current_state其他列归0或者出来的act其他列归0
                act_values = self.model(torch.cuda.FloatTensor(state)).detach().cpu().numpy()
                # 取消---act_values只对本次请求的容器做预测，其他设为0
                # act_values[np.logical_not(np.isin(np.arange(len(act_values)), np_slice_unique))] = 0
                max_index = np.argmax(act_values).item()  # 获取最大值的索引
                binary_index = np.binary_repr(max_index, width=5)  # 将索引转换为二进制向量
                predicted_actions = np.array(list(binary_index), dtype=int)
                return predicted_actions
            # 将预测结果插入到全局动作向量中

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        batch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.target_model(torch.cuda.FloatTensor(next_state)).detach().cpu().numpy())
            target_f = self.model(torch.cuda.FloatTensor(state)).detach().cpu().numpy()
            action_index = int(''.join([str(int(a)) for a in action]), 2)
            target_f[action_index] = target
            states.append(state)
            targets.append(target_f)
        self.optimizer.zero_grad()
        # loss = F.mse_loss(self.model(torch.cuda.FloatTensor(states)), torch.cuda.FloatTensor(targets))
        states = np.array(states)
        targets = np.array(targets)

        # 将numpy数组转换为PyTorch张量并将数据移动到GPU上
        states = torch.as_tensor(states, dtype=torch.float32, device=device)
        targets = torch.as_tensor(targets, dtype=torch.float32, device=device)

        # 计算损失
        predictions = self.model(states)
        loss = F.mse_loss(predictions, targets)
        loss.backward()
        self.optimizer.step()

    def target_train(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, episodes):
        total_start_cost_list = []
        total_cache_cost_list = []
        total_cost_list = []
        for e in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.replay()

            self.target_train()

            # Append cost data to lists
            total_start_cost_list.append(self.env.total_start_cost)
            total_cost_list.append(self.env.total_cost)
            cache_cost = self.env.total_cost - self.env.total_start_cost
            total_cache_cost_list.append(cache_cost)

            print("epoch:{}".format(e))
            print("total_start_cost = {}".format(self.env.total_start_cost))
            print("total_cache_cost = {}".format(cache_cost))
            print("total_cost = {}".format(self.env.total_cost))
            print("-----------------")
        # Return the cost data lists
        return total_start_cost_list, total_cache_cost_list, total_cost_list

    def test(self):
        state = self.env.reset()
        done = False
        while not done:
            # action = np.argmax(self.model(torch.FloatTensor(state)).detach().numpy())
            act_values = self.model(torch.cuda.FloatTensor(state)).detach().cpu().numpy()
            # 取消---act_values只对本次请求的容器做预测，其他设为0
            # act_values[np.logical_not(np.isin(np.arange(len(act_values)), np_slice_unique))] = 0
            max_index = np.argmax(act_values).item()  # 获取最大值的索引
            binary_index = np.binary_repr(max_index, width=5)  # 将索引转换为二进制向量
            action = np.array(list(binary_index), dtype=int)
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

    # match = re.search(r"v2/([^/]+)/([^/]+)", uri)
    # if match:
    #     uri_parts = match.groups()
    #     # uri = "/".join(uri_parts)
    #     uri = uri_parts[0]
    # else:
    #     uri = "v2"

    return [uri, timestamp]

def preprocess_data(raw_data,interval):
    # 将原始数据转换为数据帧
    df = pd.DataFrame(raw_data, columns=["container_type", "timestamp"])

    # 计算每个容器类型出现的频率
    freq = df["container_type"].value_counts()

    # 将出现频率低于阈值的类型替换为"else"
    low_freq_types = freq[freq <= 2000].index
    df["container_type"].replace(low_freq_types, "miscellaneous", inplace=True)
    low_freq_types = freq[(freq > 2000) & (freq <= 4000)].index
    df["container_type"].replace(low_freq_types, "small", inplace=True)

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
# n秒为1个时间片
interval="1S"

json_files = load_json_files(data_path)
raw_data = load_data_generator(json_files)
data = preprocess_data(raw_data,interval)

print("数据处理完成")

max_cache_num = 10  # 最大缓存容器数
alpha = 0.004  # 缓存开销系数
cache_slice_num = int(1/alpha/5)
type_num = 5
max_remain_slices = 1000  # 最大缓存时间片数
parameters_env = max_remain_slices, max_cache_num, cache_slice_num, alpha, type_num

gamma = 0.99
# epsilon = 1.0
# epsilon_min = 0.01
# epsilon_decay = 0.995
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.999 #大约690次迭代后小于0.5 每个batch_size乘一次，大约在22080次运行小于0.5
batch_size = 32
parameters_agent = gamma, epsilon, epsilon_min, epsilon_decay, batch_size

# 初始化环境和DQNAgent
env = ContainerCacheEnv(data, parameters_env)
agent = DQNAgent(env, parameters_agent)

# 训练模型
total_start_cost_list, \
total_cache_cost_list, \
total_cost_list = agent.train(10)

# 测试模型
cost = agent.test()
total_cost, start_cost, cache_cost = cost
print("------------test-------------")
print("Total cost:", total_cost)
print("Start cost:", start_cost)
print("Cache cost:", cache_cost)

# Save the trained DQNAgent model
with open('dqn_agent.pkl', 'wb') as f:
    pickle.dump(agent, f)

# Load the saved DQNAgent model
# with open('dqn_agent.pkl', 'rb') as f:
#     agent = pickle.load(f)

# Plot total_start_cost
plt.plot(total_start_cost_list)
plt.title('Total Start Cost')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()

# Plot total_cache_cost
plt.plot(total_cache_cost_list)
plt.title('Total Cache Cost')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()

# Plot total_cost
plt.plot(total_cost_list)
plt.title('Total Cost')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()
