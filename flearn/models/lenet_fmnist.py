import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    专为 28x28 图像 (如 FMNIST, MNIST) 优化的 LeNet-5
    """

    def __init__(self, in_channel=1, num_classes=10, **kwargs):
        super(LeNet5, self).__init__()

        # ---------------- 卷积特征提取层 ----------------
        self.features = nn.Sequential(
            # 接收 in_channel (FMNIST 传入 1)
            nn.Conv2d(in_channels=in_channel, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ---------------- 全连接分类层 ----------------
        self.classifier = nn.Sequential(
            # 【关键修改】：FMNIST 是 28x28，到这里变成了 4x4
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平操作
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # 模拟 FMNIST 测试
    global_model = LeNet5(in_channel=1, num_classes=10)
    # 生成一个 28x28 的单通道假数据
    dummy_input = torch.randn(2, 1, 28, 28)
    output = global_model(dummy_input)
    print("FMNIST 模型调用成功！输出维度:", output.shape)