'''
Jihyun Lim <wlguslim@inha.edu>
Inha University
2025/8/12
'''
import torch.nn as nn
import torch.nn.init as init

def _model_weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = kernel_size,
                               stride = stride,
                               padding = 1,
                               bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels = out_channels,
                               out_channels = out_channels,
                               kernel_size = kernel_size,
                               padding = 1,
                               bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        nn.init.constant_(self.bn2.weight, 0.0)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,
                          out_channels = out_channels,
                          kernel_size = 1,
                          stride = stride,
                          padding = 0,
                          bias = False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.shortcut(x)
        y = self.relu(out)
        return y

class ResNet20(nn.Module):
    def __init__(self, num_classes, ratio = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ratio = ratio

        self.conv1 = nn.Conv2d(in_channels = 3,
                               out_channels = 16,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               bias = False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.res_block1 = self._make_block(num_block = 3,
                                           in_channels = int(16 * self.ratio),
                                           out_channels = int(16 * self.ratio),
                                           stride = 1)
        self.res_block2 = self._make_block(num_block = 3,
                                           in_channels = int(16 * self.ratio),
                                           out_channels = int(32 * self.ratio),
                                           stride = 2)
        self.res_block3 = self._make_block(num_block = 3,
                                           in_channels = int(32 * self.ratio),
                                           out_channels = int(64 * self.ratio),
                                           stride = 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(int(64 * self.ratio), num_classes)
        self.apply(_model_weight_init)

    def _make_block(self, num_block, in_channels, out_channels, stride):
        blocks = []
        strides = [stride] + [1] * (num_block - 1)
        in_channels_list = [in_channels] + [out_channels] * (num_block - 1)
        
        for i in range(num_block):
            blocks.append(ResBlock(in_channels = in_channels_list[i],
                                   out_channels = out_channels, 
                                   stride = strides[i]))
        
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y

