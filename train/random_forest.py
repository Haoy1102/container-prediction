import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import os
import json
import re
import numpy as np
import time
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 记录程序开始时间
start_time = time.time()

# 将单个.json文件中的请求格式转换为数据集
def transform_data(request):
    # 提取特征
    host = request["host"]
    duration = request["http.request.duration"]
    method = request["http.request.method"]
    remoteaddr = request["http.request.remoteaddr"]
    useragent = request["http.request.useragent"]
    uri = request["http.request.uri"]
    timestamp = request["timestamp"]

    match = re.search(r"v2/([^/]+)/([^/]+)", uri)
    if match:
        uri_parts = match.groups()
        uri = "/".join(uri_parts)
    else:
        uri = "v2"

    # 获取标签
    #TODO 如果找第2个数据则100% 出问题
    label = uri.split('/')[1] if len(uri.split('/')) >= 2 else 'default'
    # if len(uri.split('/')) >= 3:
    #     label = uri.split('/')[2]
    # else:
    #     label = 'default'

    # 返回特征和标签
    return [host, duration, method, remoteaddr, useragent,uri,timestamp,label]

def preprocess_data(raw_data):
    # 将原始数据转换为数据帧
    df = pd.DataFrame(raw_data, columns=["host", "duration", "method", "remoteaddr", "useragent","uri","timestamp","label"])

    # 处理数值型特征
    num_features = ["duration"]
    num_data = df[num_features].values

    # 处理时间戳特征
    timestamp = pd.to_datetime(df["timestamp"])
    timestamp_diff = (timestamp - timestamp.min()).dt.total_seconds().values.reshape(-1, 1)

    # 处理分类型特征
    cat_features = ["host","method", "remoteaddr", "useragent"]
    cat_data = df[cat_features].values

    # 使用one-hot编码
    encoder = OneHotEncoder()
    cat_data_encoded = encoder.fit_transform(cat_data).toarray()

#---------旧的-------------------------
    # 使用label编码
    label_encoded = df["label"].values
    # 标签编码
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(label_encoded)

#---------新的------------------------
    # uri_encoded = pd.Series(df["uri"]).astype('category').cat.codes.values.reshape(-1, 1)
    # uri_encoded = df["uri"].values
    # label_encoder = LabelEncoder()
    # uri_encoded = label_encoder.fit_transform(uri_encoded)
    # # 使用kmeans算法处理uri
    # kmeans = KMeans(n_clusters=50, random_state=0)
    # uri_clustered = kmeans.fit_transform(uri_encoded)
    #
    # # 新的标签
    # label_encoded = kmeans.labels_

 #----------------------------------


    # # 将uri属性转换为TF-IDF向量
    # tfidf_vectorizer = TfidfVectorizer()
    # uri_tfidf = tfidf_vectorizer.fit_transform(df["uri"].values)
    #
    # # 使用kmeans算法处理uri
    # kmeans = KMeans(n_clusters=50, random_state=0)
    # uri_clustered = kmeans.fit_transform(uri_tfidf)
    #
    # # 新的标签
    # label_encoded = kmeans.labels_
#------------------
    # 合并特征
    features = np.concatenate((num_data,timestamp_diff, cat_data_encoded), axis=1)

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
for json_file in tqdm.tqdm(json_files, desc="Loading JSON files"):
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

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=True)

# 将数据集转换为PyTorch中的张量
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

y_train_cpu = y_train.cpu()
y_train_np = y_train_cpu.numpy()

print("数据预处理完成")

# 定义决策森林分类模型
class DecisionForest(nn.Module):
    def __init__(self, n_features, n_classes, n_trees):
        super(DecisionForest, self).__init__()
        self.n_trees = n_trees
        self.trees = nn.ModuleList([DecisionTree(n_features, n_classes) for _ in range(n_trees)])

    def forward(self, x):
        output = 0
        for tree in self.trees:
            output += tree(x)
        return output / self.n_trees

# 定义单棵决策树模型
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

def test(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        y_pred = torch.argmax(output, dim=1).cpu().numpy() # 将张量移动回CPU上
        acc = accuracy_score(y_test.cpu().numpy(), y_pred)
    return acc

# 定义超参数
n_features = X_train.shape[1]
n_classes = len(np.unique(y_train_np))
lr = 0.05
epochs = 100
n_trees = 6
batch_size = 1024

# 创建模型、优化器和损失函数
model = DecisionForest(n_features, n_classes, n_trees).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# 定义数据加载器
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 训练模型
for epoch in range(epochs):
    loss_total = 0
    for i, (batch_x, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()

        if (i+1) % 50 == 0:
            acc = test(model, X_test, y_test)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, epochs, i+1, len(train_loader), loss.item(), acc * 100))

    acc = test(model, X_test, y_test)
    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, epochs, loss_total / len(train_loader), acc * 100))

# 在测试集上进行评估
with torch.no_grad():
    model.eval()
    output = model(X_test)
    y_pred = torch.argmax(output, dim=1).cpu().numpy() # 将张量移动回CPU上
    acc = accuracy_score(y_test.cpu().numpy(), y_pred)
    print('Test Accuracy: {:.2f}%'.format(acc * 100))

# 记录程序结束时间
end_time = time.time()

# 计算程序运行总时间
total_time = end_time - start_time

print("程序运行总时间为：{}秒".format(total_time))

