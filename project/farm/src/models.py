from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn
import torch

class CNN_Encoder(nn.Module):
    def __init__(self, class_n, rate=0.1):
        super(CNN_Encoder, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    def forward(self, inputs):
        output = self.model(inputs)
        return output