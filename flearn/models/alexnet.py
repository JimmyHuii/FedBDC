import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    专门为小尺寸图像（如 CIFAR 32x32）修正的 AlexNet 变体。
    - 修复了原始大卷积核导致的特征图塌陷问题。
    - 引入了 BatchNorm 解决初期梯度消失/震荡问题。
    - 减小了分类器的参数量，更适合联邦学习环境。
    """

    def __init__(self, in_channel=3, num_classes=100):  # 注意: CIFAR-100 默认 num_classes=100
        super(AlexNet, self).__init__()

        # 特征提取部分（使用 3x3 卷积核适配 32x32 图像）
        self.features = nn.Sequential(
            # Input: 32x32 -> Conv -> 32x32 -> MaxPool -> 16x16
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 16x16 -> Conv -> 16x16 -> MaxPool -> 8x8
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 8x8 -> Conv -> 8x8
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # 8x8 -> Conv -> 8x8
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 8x8 -> Conv -> 8x8 -> MaxPool -> 4x4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 此时特征图大小为 256 * 4 * 4 = 4096
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 4 * 4, 1024),  # 将 4096 缩小到 1024 避免过度冗余和通信开销
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平
        x = self.classifier(x)
        return x


# 测试代码，确保跑通不报错
if __name__ == "__main__":
    net = AlexNet(num_classes=100)
    # 模拟一个 CIFAR-100 的 Batch: (batch_size, channels, height, width)
    dummy_input = torch.randn(16, 3, 32, 32)
    output = net(dummy_input)
    print(f"Output shape: {output.shape}")  # 应该输出 [16, 100]