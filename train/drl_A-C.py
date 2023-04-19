import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ActorCritic, self).__init__()

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy = F.softmax(self.actor(x), dim=1)
        value = self.critic(x)

        return policy, value


class ContainerCacheEnv(gym.Env):
    def __init__(self, data):
        self.data = data
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(15,))
        self.action_space = gym.spaces.Box(low=0, high=1000, shape=(15,))

    def reset(self):
        self.index = 0
        self.memory = np.zeros(15)
        return self.memory

    def step(self, action):
        # Update memory
        for i in range(15):
            if action[i] > 0:
                self.memory[i] = action[i]

        # Calculate reward
        reward = 0
        if sum(self.data[self.index]) > 0:
            for i in range(15):
                if self.memory[i] == 0:
                    reward += 1
            reward += np.sum(action)

        # Update index
        self.index += 1

        # Return next state and reward
        return self.memory, reward, self.index == len(self.data), {}

def train(env, model, optimizer):
    obs = env.reset()
    done = False
    while not done:
        # Sample action from policy
        policy, value = model(torch.FloatTensor(obs))
        action = policy.multinomial(num_samples=1).data[0]

        # Take step in environment
        next_obs, reward, done, _ = env.step(action.numpy())

        # Calculate advantage
        _, next_value = model(torch.FloatTensor(next_obs))
        td_target = reward + 0.99 * next_value.data[0]
        td_error = td_target - value.data[0]
        advantage = torch.FloatTensor([td_error])

        # Update actor and critic
        log_policy = torch.log(policy)
        selected_log_policy = advantage * log_policy.gather(1, torch.LongTensor(action))
        actor_loss = -selected_log_policy
        critic_loss = F.smooth_l1_loss(value, torch.FloatTensor([td_target]))
        loss = actor_loss + critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = next_obs

# Example usage
data = [[1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 100
env = ContainerCacheEnv(data)
model = ActorCritic(num_inputs=15, num_actions=15, hidden_size=128)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for i in range(1000):
    train(env, model, optimizer)
