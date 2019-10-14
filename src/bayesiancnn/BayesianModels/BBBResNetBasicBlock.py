import torch.nn as nn
from BBBlayers import BBBConv2d, BBBLinearFactorial, FlattenLayer

class BBBResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BBBResNetBasicBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.projection = self.stride != 1 or self.in_channels != self.out_channels
        identity = nn.Sequential()
        self.projcomp = self.projection_component() if self.projection_component else identity

        self.conv1 = BBBConv2d(self.in_channels, self.out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.soft1 = nn.Softplus()
        self.conv2 = BBBConv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.soft2 = nn.Softplus()

        layers = [self.conv1, self.bn1, self.soft1, self.conv2, self.bn2, self.projcomp, self.soft2]

        self.layers = nn.ModuleList(layers)

    def projection_component(self):
        conv1 = BBBConv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride)
        bn1 = nn.BatchNorm2d(self.out_channels)
        return nn.Sequential(conv1, bn1)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0

        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl = layer.fcprobforward(x, ret_mean_std=False)
                kl += _kl    
            else:
                x = layer(x)
        
        logits = x
       
        return logits, kl