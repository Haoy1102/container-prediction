import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

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


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# 定义神经网络模型类
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 设置超参数和设备类型（CPU或GPU）
input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = len(np.unique(y_train))
lrate = 0.01
num_epochs = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batchsize = 32  # 批量读取数据时每个批次包含多少条样本
# 将数据集转换为TensorDataset格式，并创建DataLoader对象用于批量读取数据
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize)
# 初始化神经网络、损失函数和优化器
model = Net(input_size=input_dim,
            hidden_size=hidden_dim,
            output_size=output_dim).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
# 训练模型
for epoch in range(num_epochs):
    for i, (inputs, target) in enumerate(train_loader):
        inputs, target = (inputs.to(device), target.to(device))
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, target.long())
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, float(loss)))
    # 模型评估
with torch.no_grad():
    correct, total = 0, len(test_loader.dataset)
    for inputs, target in test_loader:
        inputs, target = (inputs.to(device), target.to(device))
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs.data, dim=-1)
        correct += (predicted == target).sum().item()
print('Accuracy of the network on the test data: {} %'.format(100 * correct / total))
