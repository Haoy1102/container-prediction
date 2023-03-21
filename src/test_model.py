import torch
import numpy as np
from sklearn.metrics import accuracy_score
from preprocessing.data_preprocess import preprocess_data
from models.decision_tree import DecisionTree
from utils.logger import get_logger

logger = get_logger(__name__)

def test_model(X_test, y_test, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    with torch.no_grad():
        model.eval()
        output = model(X_test)
        y_pred = torch.argmax(output, dim=1).cpu().numpy() # 将张量移动回CPU上
        acc = accuracy_score(y_test.cpu().numpy(), y_pred)
        logger.info('Test Accuracy: {:.2f}%'.format(acc * 100))
    return acc
