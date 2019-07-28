import torch.nn as nn
from BBBlayers import BBBLinearFactorial

class BBBFCHousePricing(nn.Module):
    def __init__(self,inputs, bias=False):
        super(BBBFCHousePricing, self).__init__()
        self.fc1 = BBBLinearFactorial(inputs, 8, bias=bias)
        self.soft1 = nn.ReLU()

        self.fc2 = BBBLinearFactorial(8, 4, bias=bias)
        self.soft2 = nn.ReLU()

        self.fc3 = BBBLinearFactorial(8, 1, bias=bias)

        layers = [self.fc1, self.soft1,  self.fc3] #self.fc2, self.soft2,

        self.layers = nn.ModuleList(layers)
    
    def probforward(self, x, ret_mean_std=False):
      'Forward pass with Bayesian weights'
      kl = 0
      fc_qw_mean = 0
      fc_qw_std = 0
      for idx, layer in enumerate(self.layers):
          if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
              x, _kl, = layer.convprobforward(x)
              kl += _kl

          elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
              if ret_mean_std and idx == len(self.layers) - 1 :
                  x, _kl, fc_qw_mean, fc_qw_std = layer.fcprobforward(x, True)
                  kl += _kl
              else:
                  #print(layer)
                  x, _kl = layer.fcprobforward(x, ret_mean_std=False)
                  kl += _kl    
          else:
              x = layer(x)
      logits = x
      #print('logits', logits)

      if not ret_mean_std:
          return logits, kl
      else:
          return logits, kl, fc_qw_mean, fc_qw_std