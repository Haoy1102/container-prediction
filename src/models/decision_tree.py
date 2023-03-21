import torch.nn as nn

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
