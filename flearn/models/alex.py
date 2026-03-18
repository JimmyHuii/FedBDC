import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    现代化改造版 AlexNet (专为 CIFAR/FMNIST 等小图片和联邦学习优化)
    1. 使用 3x3 卷积核替代了原始的 11x11 大卷积核，以适应 32x32 分辨率。
    2. 使用 全局平均池化 (GAP) 彻底替换了庞大且容易过拟合的全连接层。
    """

    # 【修改点】将 num_channels 改为 in_channel，并加入 **kwargs 完美兼容 base.py
    def __init__(self, args=None, num_classes=10, in_channel=3, **kwargs):
        super(AlexNet, self).__init__()

        # 兼容你的联邦学习框架参数传入
        if args is not None:
            if hasattr(args, 'num_classes'):
                num_classes = args.num_classes
            if hasattr(args, 'in_channel'):
                in_channel = args.in_channel

        # ---------------- 特征提取层 (卷积部分) ----------------
        self.features = nn.Sequential(
            # Layer 1 (输入通道数改为 in_channel)
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 2
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Layer 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Layer 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ---------------- 分类器层 (现代化 GAP 改造) ----------------
        self.classifier = nn.Sequential(
            # 将每个通道 (256个通道) 的特征图无论多大，都平均成一个点 (1x1)
            nn.AdaptiveAvgPool2d((1, 1)),
            # 展平，维度变为 256
            nn.Flatten(),
            # 极轻量级的输出层：256 -> num_classes，参数量仅为 256 * 10 = 2560
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 可选：如果你的代码里是通过函数调用的，可以保留这个接口
def alexnet(args=None, **kwargs):
    return AlexNet(args, **kwargs)


if __name__ == '__main__':
    # 本地测试一下参数量和输入输出
    model = AlexNet(num_classes=10, in_channel=3)
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 应该是 (2, 10)

    # 统计分类器参数量
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    print(f"Classifier parameters count: {classifier_params}")  # 应该只有 2570 个