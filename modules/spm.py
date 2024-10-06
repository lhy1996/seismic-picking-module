import math
import numpy as np
import torch
from torch import nn
from torch.nn import init

class SPM(nn.Module):

    def __init__(self, channel=512, reduction=4,input_size=None):
        super().__init__()
        self.conv11 = nn.Conv1d(channel, 1, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Sequential(
            nn.Linear(input_size, math.ceil(input_size // reduction) ,bias=False),
            nn.ELU(inplace=True),
            nn.Linear(math.ceil(input_size // reduction), input_size, bias=False)
        )
        self.conv12 = nn.Sequential(
            nn.Conv1d(1,channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y2 = self.conv11(x)
        y2 = self.fc2(y2)
        y2 = self.conv12(y2)

        return x * y2

# if __name__ == '__main__':
#     input = torch.randn(50, 512, 3001)
#     sp = SPM(channel=512,input_size=3001)
#     output = sp(input)
#     print(output.shape)