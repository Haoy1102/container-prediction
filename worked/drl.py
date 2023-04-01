import gym
import torch.nn.functional as F

class EdgeNodeEnv(gym.Env):
    def __init__(self, data):
        self.data = data
        self.memory_size = 10  # 内存大小设为10
        self.containers_in_memory = set()  # 当前在内存中的容器集合
        self.action_space = gym.spaces.Discrete(2)  # 动作空间有两种：0表示载入新容器，1表示清除旧容器
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(data[0]) - 2,),
                                                dtype=float)  # 观测空间由数据项中除了host和timestamp以外所有属性组成

    def reset(self):
        """
        重置环境状态并返回初始观测值。
        在本例中，我们假定最初没有任何容器被加载到内存中。
        Returns:
            observation (object): 初始观察值。
        """
        self.containers_in_memory.clear()
        return self._get_observation()

    def step(self, action):
        """
        采取给定动作并返回更新后的状态、奖励、是否终止等信息。
        Args:
           action (object): 要执行的动作。
        Returns:
           observation (object): 更新后的观察值。
           reward (float) : 由于执行动作而获得的奖励。
           done (bool): 是否到达终止状态。在本例中，我们假设永远不会结束。
        """
        if action == 0:  # 载入新容器
            container_type = self._get_container_type()
            if len(self.containers_in_memory) < self.memory_size and container_type not in self.containers_in_memory:
                # 如果内存未满且该类型容器不在内存中，则将其载入内存并给予正向奖励
                self.containers_in_memory.add(container_type)
                reward = 1.0
            else:
                # 否则给予负向奖励，并保持现有状态不变
                reward = -1.0
        elif action == 1:  # 清除旧容器
            if len(self.containers_in_memory) > 0:
                containers_by_age = sorted(list(self.containers_in_memory), key=lambda x: "/".join(x.split("/")[1:3]))

                oldest_container = containers_by_age[0]

                if oldest_container in self.containers_in_memory:
                    # 将最老的容器清出内存并给予正向奖励
                    self.containers_in_memory.remove(oldest_container)
                    reward = 1.0

            else:
                # 如果没有可清除的容器，则保持现有状态不变，并给予负向奖励
                reward = -1.0
        else:
            raise ValueError("Invalid action: {}".format(action))

        print("Memory:", self.containers_in_memory)
        return self._get_observation(), reward, False, {}

    def _get_container_type(self):
        """
        从数据集中随机选择一个容器类型。
        Returns:
            container_type (str): 容器类型。
        """
        var = []
        for d in self.data:
            if "http.request.uri" in d and len(d["http.request.uri"].split("/")) >= 4:
                uri_parts = d["http.request.uri"].split("/")
                var.append(uri_parts[1] + "/" + uri_parts[2])
            else:
                var.append("default_value")
        return random.choice(var)

    def _get_observation(self):
        """
        返回当前状态的观察值。
        Returns:
           observation (object): 当前状态的观察值。
        """
        obs = []

        for key in self.data[0]:
            if key != "host" and key != "timestamp":
                obs.append(1.0 if any(key in d and d[key] is not None for d in self.data) else 0.0)

        return obs


import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
    def __init__(self, env):

        # 初始化环境和超参数

        self.env = env
        self.gamma = 0.99  # 折扣因子
        self.epsilon_start = 1.0  # 贪婪策略起始值
        self.epsilon_end = 0.01  # 贪婪策略结束值
        self.epsilon_decay = 5000  # epsilon线性衰减的步数

        # 初始化经验回放缓冲区、当前状态和epsilon值。

        self.memory_capacity = 10000
        self.memory = []

        self.current_state = None
        self.current_epsilon = self.epsilon_start

        # 将模型和优化器放置在GPU上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        # 根据当前状态选择动作。

        device = next(self.model.parameters()).device  # 获取设备信息

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

        device = next(self.model.parameters()).device  # 获取设备信息

        state = self.current_state
        action = self.act(state)

        next_state, reward, done, _ = self.env.step(action)
        # 将经验存储到经验回放缓冲区中。
        self.memory.append((state, action, reward, next_state))

        # 从经验回放缓冲区中随机抽取一批样本，并使用它们更新模型参数。
        self.replay_memory(batch_size=32)

        # 更新当前状态和epsilon值，然后返回奖励信息。
        self.current_state = next_state
        self.current_epsilon = max(self.epsilon_end,
                                   self.current_epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay)


        return reward

    def train(self, num_steps):
        total_reward = 0.0
        for i in range(num_steps):
            reward = self.train_step()
            total_reward += reward

        return total_reward


import matplotlib.pyplot as plt

import json
import os
from tqdm import tqdm


# 加载数据集并解析为JSON格式。
json_files = []
for root, dirs, files in os.walk("../dataset/node-9e"):
    for file in files:
        if file.endswith(".json"):
            json_file = os.path.join(root, file)
            json_files.append(json_file)

data = []
for json_file in tqdm(json_files):
    with open(json_file, "r") as f:
        json_object = json.load(f)
        data.extend(json_object)


env = EdgeNodeEnv(data)
agent = DQNAgent(env)
rewards = []
for i in range(100):
    agent.current_state = env._get_container_type()
    reward = agent.train(num_steps=100)
    rewards.append(reward)
    print("epoch:{}".format(i))

# plt.switch_backend('TkAgg')
plt.plot(range(len(rewards)), rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()