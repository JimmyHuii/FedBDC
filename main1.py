# coding:utf-8
from flearn.experiments.base import CentralTraining
from flearn.utils.options import args_parser
import torch


if __name__ == "__main__":
    args = args_parser()
    args.model = "alexnet"
    #
    args.dataset = "cifar100"

    if args.model == "vgg" or args.model == "alexnet":
        args.epochs = 800



    # # 1. 在一个地方定义您所有的 "固定" 优先级
    # priority_map = {
    #     'cnn': ['classifier.2', 'features.0', 'features.3', 'classifier.0', 'features.6'],
    #     'vgg': ['classifier.4', 'features.0', 'features.3', 'features.8',
    #             'features.6', 'features.11', 'features.13', 'features.16',
    #             'classifier.2', 'features.18', 'classifier.0'],
    #     'resnet': ['linear', 'conv1', 'layer2.0', 'layer1.0', 'layer1.1', 'layer1.2',
    #                'layer3.0', 'layer2.1', 'layer2.2', 'layer3.1', 'layer3.2']
    # }
    #
    # # 2. 动态地将列表 "添加" 到 args 对象中
    # #    (我们是在解析 *之后* 添加它，所以不需要 parser.add_argument)
    # args.priority_list = priority_map.get(args.model, None)
    #
    # if args.priority_list:
    #     print(f"--- 成功为模型 {args.model} 加载了自定义优先级列表 ---")
    # else:
    #     print(f"--- 未找到模型 {args.model} 的优先级列表, 将使用默认(从后向前)顺序 ---")

    # args.hypernet_comm = True

    # args.layer_wise_up = True
    # args.layer_wise_down = True
    # args.start_epoch_layer_down = 50
    # args.active_hybrid_down = True
    # args.number_of_layer = 3

    # if args.model == "cnn":
    #     args.epochs = 588
    # elif args.model == "vgg":
    #     args.epochs = 629
    # elif args.model == "resnet":
    #     args.epochs = 724

    args.l = 2

    # args.is_FedAvg = True
    # args.is_uptopk = True
    # args.final_ratio = 0.5
    # args.is_qsgd = True
    # args.hypernet_comm = True
    # args.lambda_reg = 0.01
    # args.B = 80000000

    # args.freeze_except_last = False
    # args.layer_wise = True

    # if args.model == "cnn":
    #     args.down_ratio = 0.502
    #     args.final_ratio = 0.446
    # elif args.model == "vgg":
    #     args.down_ratio = 0.463
    #     args.final_ratio = 0.497
    # elif args.model == "resnet":
    #     args.down_ratio = 0.474
    #     args.final_ratio = 0.401



    args.global_mask_frozen = 0  # 1 使用全局冻结
    # args.result_dir = "test"

    # args.local_iter = None

    # kwargs = {'num_workers': 6, 'pin_memory': False} if torch.cuda.is_available() else {}
    t = CentralTraining(args, iid=False, unequal=False, result_dir=args.result_dir)
    t.train()


