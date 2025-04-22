# model.py

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.network = models.resnet50(weights=ResNet50_Weights.DEFAULT)  # استخدم pretrained
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

# --- هنا الكائن المطلوب --- #
loaded_model = ResNet(num_classes=6)
