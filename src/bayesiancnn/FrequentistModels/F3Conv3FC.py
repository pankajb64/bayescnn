import torch.nn as nn
from utils.BBBlayers import FlattenLayer

class F3Conv3FC(nn.Module):
    """
    To train on CIFAR-10:
    https://arxiv.org/pdf/1207.0580.pdf
    """
    def __init__(self, outputs, inputs):
        super(F3Conv3FC, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inputs, 32, 5, stride=1, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, 5, stride=1, padding=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            FlattenLayer(2 * 2 * 128),
            nn.Linear(2 * 2 * 128, 1000),
            nn.Softplus(),
            nn.Linear(1000, 1000),
            nn.Softplus(),
            nn.Linear(1000, outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        print('X', x)
        return x