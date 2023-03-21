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

# 加载数据集
data = [ [0.1, 1.2, 0.9, 0.8, 0.2, 0.4, 0.3, 0.5, 1.0, 8],
         [0.5, 0.4, 0.8, 1.2, 0.3, 0.2, 0.6, 0.1, 0.9, 3],
         ...
         [0.9, 0.7, 1.0, 0.3, 0.1, 0.2, 0.4, 0.6, 0.8, 5] ]
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
test_data = [0.2, 1.0, 0.7, 0.5, 0.3, 0.4, 0.1, 0.8, 0.9]
with torch.no_grad():
    model.eval()
    output = model(torch.tensor(test_data, dtype=torch.float32).unsqueeze(0).unsqueeze(1))
    pred = torch.argmax(output, dim=1).item()
    print('预测结果为：', pred)
