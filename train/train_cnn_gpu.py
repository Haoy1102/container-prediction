import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time

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

    # 将数据转换为CNN模型需要的格式
    features_cnn = features_scaled.reshape(-1, 1, 1, features_scaled.shape[1])

    # 将数据转换为张量形式
    features_cnn = torch.tensor(features_cnn, dtype=torch.float32).to(device)

    return features_cnn, label_encoded

class CNN(nn.Module):
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def train(model, optimizer, criterion, X_train, y_train):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        y_pred = torch.argmax(output, dim=1).cpu().numpy() # 将张量移动回CPU上
        acc = accuracy_score(y_test.cpu().numpy(), y_pred)
    return acc



def train_model(X_train, y_train, X_val, y_val, n_classes, lr=0.001, n_epochs=10, batch_size=32, device='cpu'):
    # 创建模型
    model = CNN(n_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    train_losses = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    start_time = time.time()
    for epoch in range(n_epochs):
        # 训练模型
        running_loss = 0.0
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            # loss = train(model, optimizer, criterion, X_batch.to(device), y_batch.to(device))
            X_batch_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)
            y_batch_tensor = torch.tensor(y_batch, dtype=torch.long).to(device)
            loss = train(model, optimizer, criterion, X_batch_tensor, y_batch_tensor)
            running_loss += loss * X_batch.shape[0]
        train_loss = running_loss / X_train.shape[0]
        train_losses.append(train_loss)

        # 在验证集上评估模型
        val_loss = 0.0
        y_pred = []
        for i in range(0, X_val.shape[0], batch_size):
            X_batch = X_val[i:i+batch_size]
            y_batch = y_val[i:i+batch_size]
            with torch.no_grad():
                output = model(X_batch.to(device))
                loss = criterion(output, y_batch.to(device))
                val_loss += loss.item() * X_batch.shape[0]
                y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())
        val_loss /= X_val.shape[0]
        val_losses.append(val_loss)
        val_acc = accuracy_score(y_val.cpu().numpy(), y_pred)
        val_accs.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

        print(f"Epoch {epoch+1}/{n_epochs} - train loss: {train_loss:.4f} - val loss: {val_loss:.4f} - val acc: {val_acc:.4f}")

    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f}s")

    # 返回最佳模型和训练历史
    return best_model, train_losses, val_losses, val_accs

def test_model(model, X_test, y_test, device='cpu'):
    model.eval()
    with torch.no_grad():
        output = model(X_test.to(device))
        y_pred = torch.argmax(output, dim=1).cpu().numpy() # 将张量移动回CPU上
        acc = accuracy_score(y_test.cpu().numpy(), y_pred)
    return acc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
X, y = preprocess_data(raw_data)

# 划分数据集
n_samples = X.shape[0]
n_train = int(n_samples * 0.7)
n_val = int(n_samples * 0.2)
n_test = n_samples - n_train - n_val
X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

# 训练模型
n_classes = len(np.unique(y))
best_model, train_losses, val_losses, val_accs = train_model(X_train, y_train, X_val, y_val, n_classes, lr=0.001, n_epochs=10, batch_size=32, device='cuda')

# 在测试集上评估模型
test_acc = test_model(best_model, X_test, y_test, device='cuda')
print(f"Test accuracy: {test_acc:.4f}")