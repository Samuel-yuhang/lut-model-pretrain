import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from lutneuro.config import DEFAULT_CONFIG, LUTConfig
from lutneuro.ops.lut_conv2d import LUTConv2d

__all__ = ["lut_resnet20", "lut_resnet32", "lut_resnet44", "lut_resnet56", "lut_resnet110", "lut_resnet1202"]


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Add(nn.Module):
    def forward(self, x, y):
        return x + y


class LUTBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A", config: LUTConfig = DEFAULT_CONFIG):
        super().__init__()
        self.conv1 = LUTConv2d(
            in_planes,
            planes,
            kernel_size=(3, 3),
            stride=(stride, stride),
            config=config,
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = LUTConv2d(
            planes,
            planes,
            kernel_size=(3, 3),
            stride=(1, 1),
            config=config,
            padding=(1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.add = Add()
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0)
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.add(out, self.shortcut(x))
        out = self.relu(out)
        return out


class LUTResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3, config: LUTConfig = DEFAULT_CONFIG):
        super().__init__()
        self.in_planes = 16

        # self.conv1 = LUTConv2d(in_channels, 16, kernel_size=(3, 3), vec_len=3, ncentroids=128,
        #                        stride=(1, 1), padding=(1, 1), bias=False)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, config=config)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, config=config)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, config=config)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, config: LUTConfig = DEFAULT_CONFIG):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, config=config))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, int(out.size()[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def lut_resnet20(num_classes=10, config: LUTConfig = DEFAULT_CONFIG):
    return LUTResNet(LUTBasicBlock, [3, 3, 3], num_classes, config=config)


def lut_resnet32(num_classes=10, config: LUTConfig = DEFAULT_CONFIG):
    return LUTResNet(LUTBasicBlock, [5, 5, 5], num_classes, config=config)


def lut_resnet44(num_classes=10, config: LUTConfig = DEFAULT_CONFIG):
    return LUTResNet(LUTBasicBlock, [7, 7, 7], num_classes, config=config)


def lut_resnet56(num_classes=10, config: LUTConfig = DEFAULT_CONFIG):
    return LUTResNet(LUTBasicBlock, [9, 9, 9], num_classes, config=config)


def lut_resnet110(num_classes=10, config: LUTConfig = DEFAULT_CONFIG):
    return LUTResNet(LUTBasicBlock, [18, 18, 18], num_classes, config=config)


def lut_resnet1202(num_classes=10, config: LUTConfig = DEFAULT_CONFIG):
    return LUTResNet(LUTBasicBlock, [200, 200, 200], num_classes, config=config)


def test(net):
    import numpy as np

    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith("lut_resnet"):
            print(net_name)
            test(globals()[net_name]())
            print()
