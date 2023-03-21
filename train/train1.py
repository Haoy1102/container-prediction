from sklearn.preprocessing import LabelEncoder
import os
import json

# 将单个.json文件中的请求格式转换为数据集
def transform_data(request):
    # 提取特征
    duration = request["http.request.duration"]
    method = request["http.request.method"]
    remoteaddr = request["http.request.remoteaddr"]
    useragent = request["http.request.useragent"]
    # status = request["http.response.status"]
    # written = request["http.response.written"]
    # 提取标签
    label = request["http.request.uri"]
    # 标签编码
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform([label])[0]
    # 返回特征和标签
    # return [duration, method, remoteaddr, useragent, status, written, label]
    return [duration, method, remoteaddr, useragent, label]


# 将多个.json文件中的请求格式转换为数据集
def transform_dataset(json_files):
    dataset = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                requests = json.load(f)
                for request in requests:
                    data = transform_data(request)
                    dataset.append(data)
        except ValueError as value_err:
            print("ValueError: " + str(value_err) + " in JSON file: " + json_file)
        except json.JSONDecodeError as json_err:
            print("JSONDecodeError: " + str(json_err) + " in JSON file: " + json_file)
    return dataset


# 遍历所有文件夹，获取所有.json文件的路径
json_files = []
for root, dirs, files in os.walk("../dataset"):
    for file in files:
        if file.endswith(".json"):
            json_file = os.path.join(root, file)
            json_files.append(json_file)

# print(json_files)
# 将所有.json文件中的数据加入到数据集
raw_data = transform_dataset(json_files)
print("raw_data准备完成")

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(raw_data):
    # 将原始数据转换为数据帧
    df = pd.DataFrame(raw_data)

    # 处理数值型特征
    num_features = ["http.request.duration", "http.response.written"]
    num_data = df[num_features].values

    # 处理分类型特征
    cat_features = ["http.request.method", "http.request.remoteaddr", "http.request.useragent"]
    cat_data = df[cat_features].values

    # 使用one-hot编码
    encoder = OneHotEncoder()
    cat_data_encoded = encoder.fit_transform(cat_data).toarray()

    # 合并特征
    features = np.concatenate((num_data, cat_data_encoded), axis=1)

    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled

features = preprocess_data(raw_data)


# print(dataset)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 定义数据集类
class RequestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 将数据转换为tensor
        features = torch.tensor(self.data[idx][:-1], dtype=torch.float32)
        label = torch.tensor(self.data[idx][-1], dtype=torch.long)
        return features, label

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32*8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 32*8)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

import random

def split_dataset(dataset, split_ratio=0.7):
    train_set_size = int(len(dataset) * split_ratio)
    random.shuffle(dataset)
    train_set = dataset[:train_set_size]
    test_set = dataset[train_set_size:]
    return train_set, test_set

# dataset = dataset  # 原始数据集
train_set, test_set = split_dataset(dataset, split_ratio=0.8)
print("训练集大小：", len(train_set))
print("测试集大小：", len(test_set))

# 加载数据集
data = train_set
train_data = RequestDataset(data)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 初始化模型和优化器
model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(10):
    for features, label in train_loader:
        optimizer.zero_grad()
        output = model(features.unsqueeze(1))
        loss = nn.functional.cross_entropy(output, label)
        loss.backward()
        optimizer.step()

# 预测新数据
test_data = test_set
with torch.no_grad():
    model.eval()
    output = model(torch.tensor(test_data, dtype=torch.float32).unsqueeze(0).unsqueeze(1))
    pred = torch.argmax(output, dim=1).item()
    print('预测结果为：', pred)
