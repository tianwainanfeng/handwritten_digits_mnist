import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config_loader import CONFIG

# ==== Model Architecture: Simple CNN ====

class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.dropout_prob = CONFIG["model"]["dropout"] # fractional neurons will be ignored or inactive for each training iteration to avoid overfitting, heavily relying on any single neuron

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # First conv layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Second conv layer

        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer (Flatten 7x7 feature maps to 128 nodes)
        self.fc2 = nn.Linear(128, 10)  # Output layer (10 classes for MNIST)

        self.dropout = nn.Dropout(p=self.dropout_prob)  # Dropout layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Max pooling layer with 2x2 kernel
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Max pooling

        x = x.view(x.size(0), -1) # Flatten the feature maps
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout before final layer
        x = self.fc2(x)

        return x
