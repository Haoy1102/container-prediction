import torch
import torch.nn as nn
import torch.optim as optim

# 检查是否有可用的GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 创建模型
model = MyModel()

# 将模型转移到GPU上
model.to(device)

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 将优化器转移到GPU上
optimizer.to(device)

# 定义数据
data = torch.randn(16, 10)

# 将数据转移到GPU上
data = data.to(device)

# 前向传播
output = model(data)

# 反向传播
loss = output.sum()
loss.backward()

# 更新参数
optimizer.step()
