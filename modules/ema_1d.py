import torch
import torch.nn as nn

class EMA(nn.Module):
    #Ouyang D, He S, Zhang G, et al. Efficient multi-scale attention module with cross-spatial learning[C]//ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023: 1-5.
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels % self.groups == 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool1d(1)
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv1d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv1d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h = x.size()
        group_x = x.reshape(b * self.groups, -1, h)  #
        x_h = torch.mean(group_x, dim=2, keepdim=True)
        hw = self.conv1x1(x_h)
        x1 = self.gn(group_x * torch.sigmoid(hw) * torch.sigmoid(hw.permute(0, 1, 2)))
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h)
        return (group_x * torch.sigmoid(weights)).reshape(b, c, h)

# if __name__ == '__main__':
#     block = EMA(64)
#     input = torch.rand(1, 64, 64)
#     output = block(input)
#     print(input.size(), output.size())
