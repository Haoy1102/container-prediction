import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from preprocessing.data_preprocess import preprocess_data
from models.decision_tree import DecisionTree
from utils.logger import get_logger

logger = get_logger(__name__)

def train(model, optimizer, criterion, X_train, y_train):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_model(X_train, X_test, y_train, y_test, epochs=1000, lr=0.05):
    # 将数据集转换为PyTorch中的张量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # 定义超参数
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train.cpu().numpy()))

    # 创建模型、优化器和损失函数
    model = DecisionTree(n_features, n_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    for epoch in range(epochs):
        loss = train(model, optimizer, criterion, X_train, y_train)
        acc = test(model, X_test, y_test)
        logger.info('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, epochs, loss, acc * 100))

    return model

def test(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        y_pred = torch.argmax(output, dim=1).cpu().numpy() # 将张量移动回CPU上
        acc = accuracy_score(y_test.cpu().numpy(), y_pred)
    return acc
