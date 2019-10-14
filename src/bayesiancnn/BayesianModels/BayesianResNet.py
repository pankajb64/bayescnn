import torch.nn as nn
from BBBlayers import BBBConv2d, BBBLinearFactorial, FlattenLayer
from BayesianModels.BBBResNetBasicBlock import BBBResNetBasicBlock

class BBBResNet(nn.Module):
    '''The architecture of ResNet with Bayesian Layers.
       This is a simpler, trimmed-down version of the architecture implemented 
       in PyTorch https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
       based on the original paper https://arxiv.org/pdf/1512.03385.pdf
       But it should allow us to build ResNet models with 18 and 34 layers.
    '''
    def __init__(self, outputs, inputs, block_counts):
        super(BBBResNet, self).__init__()

        if len(block_counts) != 4:
            raise ValueError('Must specify exactly four block counts')

        self.conv1 = BBBConv2d(inputs, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.comp1 = self.create_component(block_counts[0], 64, 64)
        self.comp2 = self.create_component(block_counts[1], 64, 128, stride=2)
        self.comp3 = self.create_component(block_counts[2], 128, 256, stride=2)
        self.comp4 = self.create_component(block_counts[3], 256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = FlattenLayer(512)
        self.fc = BBBLinearFactorial(512, outputs)

        layers = [self.conv1, self.bn1, self.soft1, self.pool1, self.comp1, 
        self.comp2, self.comp3, self.comp4, self.avgpool, self.flatten, self.fc]

        self.layers = nn.ModuleList(layers)

        self.reset_parameters()


    def reset_parameters(self):
        #Consider addining Kaiming Normal Initialization to conv layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def create_component(self, n_blocks, in_channels, out_channels, stride=1):
        #the basic block will need to account for projection when stride > 1 and in_channels != out_channels
        block_with_stride = BBBResNetBasicBlock(in_channels, out_channels, stride)
        layers = [block_with_stride]

        for _ in range(1, n_blocks):
            #set input and output to out_channels 
            block_no_stride = BBBResNetBasicBlock(out_channels, out_channels)
            layers.append(block_no_stride)

        return nn.Sequential(*layers)

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

    def resnet18(outputs, inputs):
        return BBBResNet(outputs, inputs, [2, 2, 2, 2])
        









