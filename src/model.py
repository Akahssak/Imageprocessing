"""
Optional CNN model for classifying segmented object patches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SimpleCNN(nn.Module):
    """
    A simple CNN classifier for object patches.
    """
    def __init__(self, num_classes: int = 2, input_size: int = 256):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Calculate the size after two pooling layers
        pooled_size = input_size // 4  # two times pooling with kernel 2
        self.fc1 = nn.Linear(32 * pooled_size * pooled_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
