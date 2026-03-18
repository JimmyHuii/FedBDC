import torch
from torch import nn
import numpy as np
import random
import torchstat

class CNN(nn.Module):

    def __init__(self, in_channel=3, num_classes=10, config=None, use_batchnorm=False):#?what is config
        super(CNN, self).__init__()
        self.in_channel = in_channel
        self.batch_norm = use_batchnorm #?

        if config is None:
            self.config = [32, 'M', 64, 'M', 64] #?
        else:
            self.config = config

        self.features = self._make_feature_layers()
        self.classifier = nn.Sequential( # 参数会按照我们定义好的序列自动传递下去
            nn.Linear(4 * 4 * self.config[-1], 64), # 定义一个神经网络的线性层（输入的神经元个数，输出的神经元个数，是否偏置） 在卷积网络中执行完最后一步张量大小为64*4*4
            nn.ReLU(inplace=True), # 激活函数，inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
            nn.Linear(64, num_classes) # 输出10个数据，因为cifar10有10种数据类型
        )


    def _make_feature_layers(self):
        layers = []
        in_channels = self.in_channel
        for param in self.config:
            if param == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2))
            else:
                layers.extend([nn.Conv2d(in_channels, param, kernel_size=3, padding=0), #卷积操作
                               nn.ReLU(inplace=True)])
                in_channels = param

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) #x.view[参数，-1]就是根据参数来调整行数,x.view[-1，参数]就是根据参数来调整列数
        x = self.classifier(x)
        return x

if __name__=="__main__":
    # 设置随机种子
    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)
    np.random.seed(777)
    random.seed(777)
    torch.backends.cudnn.deterministic = True

    model = CNN()
    torchstat.stat(model, (3, 32, 32))

    inputs = torch.ones((64, 3, 32, 32))
    outputs = model(inputs)
    print(outputs.shape)