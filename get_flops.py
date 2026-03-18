import torch
from thop import profile

from flearn.models import cnn, vgg, resnet, resnet9, resnet20, alexnet, alex, lenet, lenet_fmnist

cnn_config = [32, 'M', 64, 'M', 64]
vgg_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
resnet_config = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
lenet_config = [6, 16]

# 给定模型结构，计算模型的参数量和计算量
def get_flops(model="vgg", config=None, dataset="cifar10"):
    """
    给定模型和通道数量，返回计算量和参数量
    :param model:
    :param config:
    :return:
    """
    if dataset == "cifar10":
        in_channel = 3
        num_classes = 10
    elif dataset == "mnist" or dataset == "fmnist":
        in_channel = 1
        num_classes = 10
    elif dataset == "cifar100":
        in_channel = 3
        num_classes = 100
    else:
        exit('Error: unrecognized dataset')
    if config is None:
        print(f"using default {model} config")
    if model == "vgg" and (config is None or len(config) == len(vgg_config)):
        model = vgg.VGG11(in_channel=in_channel, num_classes=num_classes, config=config)
    elif model == "cnn" and (config is None or len(config) == len(cnn_config)):
        model = cnn.CNN(in_channel=in_channel, num_classes=num_classes, config=config)
    elif model == "resnet9":
        # model = resnet.ResNet18(num_classes=num_classes)
        model = resnet9.ResNet9(num_classes=num_classes)
    elif model == "resnet20":
        model = resnet20.ResNet20(in_channel=in_channel, num_classes=num_classes)
    elif model == "alexnet":
        model = alexnet.AlexNet(in_channel=in_channel, num_classes=num_classes)
    elif model == "alex":
        model = alex.AlexNet(in_channel=in_channel, num_classes=num_classes)
    elif model == "lenet":
        model = lenet.LeNet5(in_channel=in_channel, num_classes=num_classes)
    elif model == "lenet_fmnist":
        model = lenet_fmnist.LeNet5(in_channel=in_channel, num_classes=num_classes)
    else:

        print("unknown model")
        return 0, 0

    if dataset == "fmnist" or dataset == "mnist":
        input_image = torch.randn(1, in_channel, 28, 28)
    else:
        input_image = torch.randn(1, in_channel, 32, 32)
        
    flops, params = profile(model, inputs=(input_image,))
    return flops, params

def get_layers_name(model="vgg"):
    if model == "vgg":
        model = vgg.VGG11()
    elif model == "cnn":
        model = cnn.CNN()
    # elif model == 'resnet':
    #     model = resnet9.ResNet9()
    else:
        print("unknown model")
        return []
    layers_name = []
    for key in model.state_dict().keys():
        if "weight" in key:
            layers_name.append(key + "_and_bias")
    return layers_name

if __name__=="__main__":
    # cnn_test = [16, "M", 34, "M", 29]
    # print(get_flops("cnn", dataset="cifar10"))
    # print(get_flops("cnn", dataset="cifar100"))
    # print(get_flops("cnn", cnn_test, dataset="cifar10"))
    # vgg_test = [31, 'M', 61, 'M', 133, 130, 'M', 246, 252, 'M', 328, 247, 'M']
    # print(get_flops("vgg"))
    # print(get_flops("vgg", vgg_test))
    #
    # resnet_test = [26, 26, 26, 26, 26, 52, 52, 52, 52, 103, 103, 103, 103, 205, 205, 205, 205]
    # print(get_flops("resnet", dataset="cifar10"))
    # print(get_flops("resnet", resnet_test, dataset="cifar10"))
    #
    # lenet_test = [5, 8]
    # print(get_flops("lenet", dataset="cifar10"))
    # print(get_flops("lenet", lenet_test, dataset="cifar10"))
    print(get_layers_name("cnn"))
    flops, params = get_flops("cnn")
    print(float(flops))
    print(float(params))