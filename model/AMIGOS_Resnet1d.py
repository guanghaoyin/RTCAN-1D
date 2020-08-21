import torch.nn as nn
import math
import torch

def conv(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding = 1, bias=False)
class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out+residual
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):
    def __init__(self, input_channel, use_sub = False,layers=[2, 2, 2, 2], num_classes=10):
        super(ResNet1D, self).__init__()
        self.inplanes3 = 64
        self.use_sub = use_sub
        if use_sub == False:
            self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm1d(64)
            self.relu = nn.ReLU(inplace=True)
        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 64, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 64, layers[2], stride=2)
        self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 64, layers[3], stride=2)

        self.final_downsample = nn.Sequential(
            nn.Conv1d(self.inplanes3, 64 * BasicBlock3x3.expansion,
                      kernel_size=1, stride=3, bias=False),
            nn.BatchNorm1d(64 * BasicBlock3x3.expansion),
        )
        self.maxpool3 = nn.AvgPool1d(kernel_size=14, stride=1, padding=0)

        self.fc = nn.Linear(256, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0]* m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_sub == False:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.layer3x3_1(x)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        x = self.layer3x3_4(x)

        x = self.final_downsample(x)
        x = self.maxpool3(x)
        return x