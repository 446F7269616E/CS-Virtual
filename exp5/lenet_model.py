import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.clf = nn.Linear(84, 10)

    def feature_maps(self, x):
        feature1 = torch.sigmoid(self.conv1(x))
        pooled1 = F.avg_pool2d(feature1, kernel_size=2, stride=2)
        feature2 = torch.sigmoid(self.conv2(pooled1))
        return feature1, feature2

    def forward(self, x):
        _, feature2 = self.feature_maps(x)
        x = F.avg_pool2d(feature2, kernel_size=2, stride=2)
        x = torch.flatten(x, start_dim=1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return self.clf(x)
