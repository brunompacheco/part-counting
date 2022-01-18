"""Network definitions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()

        # two basic convolution layers and two fully-connected for regression
        self.conv1 = nn.Conv2d(2, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(1048576, 128)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Pass data through dropout1
        x = self.dropout1(x)

        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x
