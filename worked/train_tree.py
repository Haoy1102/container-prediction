import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time

# 记录程序开始时间
start_time = time.time()

# 将单个.json文件中的请求格式转换为数据集
def transform_data(request):
    # 提取特征
    duration = request["http.request.duration"]
    method = request["http.request.method"]
    remoteaddr = request["http.request.remoteaddr"]
    useragent = request["http.request.useragent"]
    uri = request["http.request.uri"]
    # 获取标签
    label = uri.split('/')[1] if len(uri.split('/')) >= 2 else 'default'
    # 返回特征和标签
    return [duration, method, remoteaddr, useragent, label]

def preprocess_data(raw_data):
    # 将原始数据转换为数据帧
    df = pd.DataFrame(raw_data, columns=["http.request.duration", "http.request.method", "http.request.remoteaddr", "http.request.useragent", "label"])

    # 处理数值型特征
    num_features = ["http.request.duration"]
    num_data = df[num_features].values

    # 处理分类型特征
    cat_features = ["http.request.method", "http.request.remoteaddr", "http.request.useragent"]
    cat_data = df[cat_features].values

    # 使用label编码
    label_encoded = df["label"].values
    # 标签编码
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(label_encoded)

    # 使用one-hot编码
    encoder = OneHotEncoder()
    cat_data_encoded = encoder.fit_transform(cat_data).toarray()

    # 合并特征
    features = np.concatenate((num_data, cat_data_encoded), axis=1)

    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, label_encoded



# 加载数据集
json_files = []
for root, dirs, files in os.walk("../dataset"):
    for file in files:
        if file.endswith(".json"):
            json_file = os.path.join(root, file)
            json_files.append(json_file)

raw_data = []
for json_file in json_files:
    with open(json_file, 'r') as f:
        try:
            requests = json.load(f)
            for request in requests:
                data = transform_data(request)
                raw_data.append(data)
        except ValueError as value_err:
            print("ValueError: " + str(value_err) + " in JSON file: " + json_file)
        except json.JSONDecodeError as json_err:
            print("JSONDecodeError: " + str(json_err) + " in JSON file: " + json_file)

features, labels = preprocess_data(raw_data)

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 将数据集转换为PyTorch中的张量
import torch

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print("数据预处理完成")

# 定义决策树分类模型
class DecisionTree(nn.Module):
    def __init__(self, n_features, n_classes):
        super(DecisionTree, self).__init__()
        self.tree = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = self.tree(x)
        return x

# 定义训练函数
def train(model, optimizer, criterion, X_train, y_train):
    model.train()
    optimizer.zero_grad()
    # output = model(torch.tensor(X_train).float().detach())
    output = model(X_train.clone().detach())
    # loss = criterion(output, torch.tensor(y_train).long())
    loss = criterion(output, y_train.clone().detach())
    loss.backward()
    optimizer.step()
    return loss.item()


# 定义测试函数
def test(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        # output = model(torch.tensor(X_test).float())
        output = model(X_test.clone().detach())
        y_pred = torch.argmax(output, dim=1).numpy()
        acc = accuracy_score(y_test, y_pred)
    return acc


# 定义超参数
n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))
lr = 0.01
epochs = 1000

# 创建模型、优化器和损失函数
model = DecisionTree(n_features, n_classes)
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# 训练模型
for epoch in range(epochs):
    loss = train(model, optimizer, criterion, X_train, y_train)
    acc = test(model, X_test, y_test)
    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, epochs, loss, acc * 100))

# 记录程序结束时间
end_time = time.time()

# 计算程序运行总时间
total_time = end_time - start_time

print("程序运行总时间为：{}秒".format(total_time))