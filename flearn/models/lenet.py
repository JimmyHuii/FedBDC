import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    专为 32x32 图像 (如 CIFAR-10) 优化的 LeNet-5
    完美兼容联邦学习框架的调用规则
    """

    def __init__(self, in_channel=3, num_classes=10, **kwargs):
        super(LeNet5, self).__init__()

        # ---------------- 卷积特征提取层 ----------------
        self.features = nn.Sequential(
            # 动态接收 in_channel (CIFAR10 传入 3)
            nn.Conv2d(in_channels=in_channel, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ---------------- 全连接分类层 ----------------
        self.classifier = nn.Sequential(
            # 注意：这里的 16*5*5 假定了输入图片的尺寸是 32x32
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            # 动态接收 num_classes (CIFAR10 传入 10)
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平操作：将 (Batch, 16, 5, 5) 变为 (Batch, 400)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # 模拟您的调用规则测试一下，确保不会报错
    in_channel = 3
    num_classes = 10

    # 即使框架额外传了别的参数（比如 args），有了 **kwargs 也不会报错
    global_model = LeNet5(in_channel=in_channel, num_classes=num_classes, extra_arg="test")

    dummy_input = torch.randn(2, 3, 32, 32)
    output = global_model(dummy_input)
    print("模型调用成功！输出维度:", output.shape)