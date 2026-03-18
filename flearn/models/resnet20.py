import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class BasicBlock(nn.Module):
    """
    ResNet 的基本残差块 (Basic Residual Block)
    """

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 快捷连接 (Shortcut Connection)
        self.shortcut = nn.Sequential()
        # 如果维度或步长不匹配，需要调整快捷连接
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 添加快捷连接
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    """
    ResNet-20 模型.
    专为CIFAR-10/100设计 (6n+2 层, n=3).
    """

    def __init__(self, in_channel=3, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_planes = 16  # ResNet-20 的初始通道数

        # 1. 初始卷积层
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 2. 三个阶段的残差块
        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)  # 3个块, 16通道
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)  # 3个块, 32通道
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)  # 3个块, 64通道

        # 3. 全连接层
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        用于构建一个阶段的残差块 (e.g., layer1, layer2, layer3)
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes  # 更新下一块的输入通道
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 初始层
        out = self.layer1(out)  # 阶段1
        out = self.layer2(out)  # 阶段2
        out = self.layer3(out)  # 阶段3

        # --- 关键修改点 ---
        # 使用自适应池化，而不是硬编码的 F.avg_pool2d(out, 4) 或 F.avg_pool2d(out, 8)
        # 这使得模型对输入图像大小 (如 32x32 或 64x64) 具有鲁棒性
        out = F.adaptive_avg_pool2d(out, (1, 1))
        # ------------------

        out = out.view(out.size(0), -1)  # 展平
        out = self.linear(out)  # 全连接层
        return out

if __name__ == "__main__":
    # 设置随机种子
    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    model = ResNet20()
    inputs = torch.ones((64, 3, 32, 32))
    outputs = model(inputs)
    # print(outputs.shape)
    model = ResNet20(3, 10)
    # stat(model, (3, 32, 32))
    print(model.state_dict())
    # input_image = torch.randn(1, 3, 32, 32)
    # flops, params = profile(model, inputs=(input_image,))
    # print(flops)
    # print(params)
    # rank = [torch.linspace(1, 64, steps=64)]
    # _, ind = model.unstructured_by_rank(rank, 0.6, 0, "cpu")
    # rank = [0, torch.linspace(1, 128, steps=128)]
    # _, ind = model.unstructured_by_rank(rank, 0.6, 1, "cpu")
    # outputs = model(inputs)
    # channels = model.get_channels()


