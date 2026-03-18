# import torch
# from torch import nn
# import numpy as np
# import random
# from torchstat import stat
#
# class VGG11(nn.Module):
#     def __init__(self, in_channel=3, num_classes=10, config=None, use_batchnorm=False):
#         super(VGG11, self).__init__()
#         self.in_channel = in_channel
#         self.batch_norm = use_batchnorm
#
#         if config is None:
#             self.config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
#         else:
#             self.config = config
#
#         self.features = self._make_feature_layers()
#         self.classifier = nn.Sequential(
#             nn.Linear(self.config[-2], 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, num_classes)
#         )
#
#     def _make_feature_layers(self):
#         layers = []
#         in_channels = self.in_channel
#         for param in self.config:
#             if param == 'M':
#                 layers.append(nn.MaxPool2d(kernel_size=2))
#             else:
#                 layers.extend([nn.Conv2d(in_channels, param, kernel_size=3, padding=1),
#                                nn.ReLU(inplace=True)])
#                 in_channels = param
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x
#
#
# if __name__=="__main__":
#     # 设置随机种子
#     seed = 777
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
#     model = VGG11()
#     inputs = torch.ones((64, 3, 32, 32))
#     outputs = model(inputs)
#     # print(outputs.shape)
#     model = VGG11(3, 10)
#     # stat(model, (3, 32, 32))
#     print(model.state_dict())
#     # input_image = torch.randn(1, 3, 32, 32)
#     # flops, params = profile(model, inputs=(input_image,))
#     # print(flops)
#     # print(params)
#     # rank = [torch.linspace(1, 64, steps=64)]
#     # _, ind = model.unstructured_by_rank(rank, 0.6, 0, "cpu")
#     # rank = [0, torch.linspace(1, 128, steps=128)]
#     # _, ind = model.unstructured_by_rank(rank, 0.6, 1, "cpu")
#     # outputs = model(inputs)
#     # channels = model.get_channels()

import torch
from torch import nn
import numpy as np
import random


class VGG11(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, config=None, use_batchnorm=True):
        super(VGG11, self).__init__()
        self.in_channel = in_channel
        # 默认开启 BatchNorm，对联邦学习极其重要
        self.batch_norm = use_batchnorm

        if config is None:
            self.config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        else:
            self.config = config

        self.features = self._make_feature_layers()

        # 此时特征图大小为 512 * 1 * 1 = 512
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # 加入 Dropout 防止过拟合
            nn.Linear(self.config[-2], 512),
            nn.BatchNorm1d(512),  # 在全连接层也加入 BN 进一步稳定训练
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def _make_feature_layers(self):
        layers = []
        in_channels = self.in_channel
        for param in self.config:
            if param == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, param, kernel_size=3, padding=1)
                # ！！！修复核心：真正把 BatchNorm 加到网络结构里 ！！！
                if self.batch_norm:
                    layers.extend([conv2d, nn.BatchNorm2d(param), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv2d, nn.ReLU(inplace=True)])
                in_channels = param

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


