import torch
import numpy as np
import random

from flearn.utils.model_util import test_inference, average_weights, get_model_like_tensor, update_model_stability, \
    calc_num_stable_params, update_model_stability_ignore_frozen, eval_avg_P_layers, update_model_stability_ignore_frozen_by_ratio, \
    average_weights_new, scale_weights, calculate_sum, replace_values, topK, recover_model_from_mask, topK_layerwise, calculate_js_divergence_for_models, initialize_param_tracker, \
    print_state_dict_keys, topK_layerwise_new, inspect_model_keys, _get_block_priority, topK_custom_hybrid, get_compressible_layers
from flearn.utils.update_include_hlw import LocalUpdate
from flearn.utils.options import args_parser
from flearn.utils.util import record_log, save_result
from flearn.models import cnn, vgg, resnet18, resnet, resnet9, resnet20, hypernet, alexnet , alex, lenet
from flearn.models.hypernet import HyperNetController, get_model_blocks
import copy
import os
import time
import cvxpy as cp


class CentralTraining(object):
    """
    对于聚合后的模型，进行中心化的训练，share_percent 是共享数据集的大小
    """

    def __init__(self, args, iid=True, unequal=False, result_dir="central"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args

        # 设置随机种子
        self.reset_seed()

        # 数据集划分信息
        self.num_data = 50000  # 数据集中前 50000 用于分配给客户端进行训练
        self.l = self.args.l  # noniid的程度， l越小 noniid 程度越大， 当 l = 1 时， 就是将数据按照顺序分成 clients 份，每个设备得到一份， 基本上只包含一个数字

        # 定义FedAvg的一些参数
        self.m = 10  # 每次取m个客户端进行平均
        self.iid = iid #True
        self.unequal = unequal #？
        self.result_dir = result_dir

        self.lr_decay = 0.99 #学习率衰减
        self.init_lr = self.args.lr

    def reset_seed(self):
        # 重置随机种子
        torch.manual_seed(self.args.seed) #为CPU中设置种子，生成随机数
        torch.cuda.manual_seed_all(self.args.seed) #为所有GPU设置种子，生成随机数 #torch.cuda.manual_seed
        np.random.seed(self.args.seed) #
        random.seed(self.args.seed)
        torch.backends.cudnn.deterministic = True #每次返回的卷积算法将是确定的，即默认算法

    def init_data(self):
        if self.args.dataset == "cifar10": #数据集为cifar10
            from data.cifar10.cifar10_data import get_dataset
            self.num_data = 50000
            train_dataset, test_dataset, user_groups = \
                get_dataset(num_data=self.num_data, num_users=self.args.num_users, iid=self.iid,
                            l=self.l, unequal=self.unequal)

        elif self.args.dataset == "cifar100":  # <-- 新增的分支
            # 从 cifar100 路径导入
            from data.cifar100.cifar100_data import get_dataset
            self.num_data = 50000  # CIFAR-100 训练集也是 50000
            train_dataset, test_dataset, user_groups = \
                get_dataset(num_data=self.num_data, num_users=self.args.num_users, iid=self.iid,
                            l=self.l, unequal=self.unequal)

        else:
            exit('Error: unrecognized dataset')

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.user_groups = user_groups  # 字典
        return train_dataset, test_dataset, user_groups



    def load_model(self):
        # BUILD MODEL
        # Convolutional neural network
        if self.args.dataset == "cifar10":
            in_channel = 3
            num_classes = 10    # 十个类别
        elif self.args.dataset == "mnist" or self.args.dataset == "fashionmnist":
            in_channel = 1
            num_classes = 10
        elif self.args.dataset == "cifar100":
            in_channel = 3
            num_classes = 100
        else:
            exit('Error: unrecognized dataset')

        if self.args.model == "cnn":
            global_model = cnn.CNN(in_channel=in_channel, num_classes=num_classes)
        elif self.args.model == "vgg":
            global_model = vgg.VGG11(in_channel=in_channel, num_classes=num_classes)
        elif self.args.model == "resnet":
            # global_model = resnet.ResNet18(num_classes=num_classes)
            # global_model = resnet9.ResNet9(num_classes=num_classes)
            global_model = resnet20.ResNet20(in_channel=in_channel, num_classes=num_classes)
        elif self.args.model == "alexnet":
            global_model = alexnet.AlexNet(in_channel=in_channel, num_classes=num_classes)
        elif self.args.model == "alex":
            global_model = alex.AlexNet(in_channel=in_channel, num_classes=num_classes)
        elif self.args.model == "lenet":
            global_model = lenet.LeNet5(in_channel=in_channel, num_classes=num_classes)
        else:
            exit('Error: unrecognized model')

        self.load_model_info(global_model)  # 无

        # Set the model to train and send it to device.
        global_model.to(self.device)
        global_model.train()    # 设置成train模式
        return global_model

    # def init_hypernetwork(self):
    #     """
    #     [新增函数] 初始化能效感知 HyperNetwork
    #     """
    #     print(">>> Initializing Energy-Aware HyperNetwork...")
    #
    #     # 1. 自动分析主模型，获取需要管理的层列表
    #     self.compressible_layers = get_compressible_layers(self.global_model)
    #     self.num_compressible = len(self.compressible_layers)
    #
    #     print(f"Blocks ({self.num_compressible}): {self.compressible_layers}")
    #
    #     print(f"    - Detected {self.num_compressible} compressible layers (Conv/Linear weights).")
    #     print(f"    - BatchNorm and Biases will be preserved (ratio=1.0).")
    #
    #     # 2. 实例化 HyperNetwork
    #     self.hypernet = hypernet.E_HyperNet(num_layers=self.num_compressible).to(self.device)
    #
    #     # 3. 定义优化器
    #     # HyperNet 的参数量很小，通常可以使用稍微不同的学习率
    #     self.hypernet_optim = torch.optim.Adam(self.hypernet.parameters(), lr=1e-3)
    #
    #     # 4. 准备层索引张量 (缓存起来，以后每次前向传播都要用)
    #     self.layer_indices = torch.arange(self.num_compressible, dtype=torch.long).to(self.device)
    #
    #     print(">>> HyperNetwork initialized successfully.")

    def load_model_info(self, model):
        """
        加载模型信息，确定可剪枝层的索引，并按照参数量进行降序排序
        :param model:
        :return:
        """
        pass

    def record_base_message(self, log_path):
        record_log(self.args, log_path, "=== " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " ===\n")
        record_log(self.args, log_path, f"=== model: {self.args.model} ===\n")

        record_log(self.args, log_path,
                   f"=== local_bs/local_ep/epochs: {self.args.local_bs}/{self.args.local_ep}/{self.args.epochs} ===\n")
    #   local_bs本地训练批量 local_ep本地训练量
    def print_info(self, user_groups=None):
        if user_groups is None:
            user_groups = [[]]
        print(f"data name: {self.args.dataset}")
        print(f"=== model: {self.args.model} ===\n")
        print(f"user nums: {self.args.num_users}")
        print(f"{'iid' if self.iid else 'noniid'} user sample nums: {len(user_groups[0])}")
        print(f"=== local_bs/local_ep/epochs: {self.args.local_bs}/{self.args.local_ep}/{self.args.epochs} ===") #local_bs本地批大小local_ep本地epoch
        print(f"=== using device {self.device} optim {self.args.optim} ===") #optim优化器

    def get_loss(self, all_trian_data, train_dataset, global_model):
        """
        获取所有训练数据的 loss
        :param user_groups:
        :param train_dataset:
        :param global_model:
        :return:
        """
        # losses = []
        # for idx in range(len(user_groups)):
        #     if user_groups[idx].shape[0] == 0:
        #         continue
        #     local_model = LocalUpdate(args=self.args, dataset=train_dataset,
        #                               idxs=user_groups[idx], device=self.device)
        #     acc, loss = local_model.inference(global_model)
        #     losses.append(loss)
        # loss = sum(losses) / len(losses)
        local_model = LocalUpdate(args=self.args, local_bs=128, dataset=train_dataset, #？ what is idxs local_bs
                                  idxs=all_trian_data, device=self.device)
        acc, loss = local_model.inference(global_model)
        return round(loss, 4) #四舍五入保留至小数点后四位

    def client_train(self, idxs_users, global_model, user_groups, epoch, train_dataset, train_losses, local_weights,
                     local_losses, mask, all_train_weights, all_train_error):
        """
        进行客户端训练
        :param idxs_users:
        :param global_model:
        :param user_groups:
        :param epoch:
        :param train_dataset:
        :param train_losses:
        :param local_weights:
        :param local_losses:
        :return:
        """
        num_current = 0
        for idx in idxs_users:
            num_current += len(user_groups[idx])
        start = time.time()
        self.args.lr = self.init_lr * pow(self.lr_decay, epoch)  # 学习率衰退
        avg_iter = (num_current * self.args.local_ep) / (self.m * self.args.local_bs)

        g_model = copy.deepcopy(global_model)
        # new_model = copy.deepcopy(global_model)

        for idx in idxs_users:
            layers_diff_epochs = []

            local_model = LocalUpdate(args=self.args, dataset=train_dataset,
                                      idxs=user_groups[idx], device=self.device, iter=self.args.local_iter)

            w, loss, error = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch,
                layers_diff_epochs=layers_diff_epochs, global_mask=mask, error=all_train_error[idx], all_train_weights=all_train_weights[idx])




            if loss < train_losses[0] * 100:    # 这个比较？
                local_weights.append([len(user_groups[idx]), copy.deepcopy(w)])
                if len(all_train_weights[idx]) == 0:
                    # all_train_weights[idx].append(g_model.state_dict())
                    all_train_weights[idx].append(copy.deepcopy(w))
                else:
                    # all_train_weights[idx][-1] = copy.deepcopy(g_model.state_dict())
                    all_train_weights[idx][-1] = copy.deepcopy(w)
                all_train_error[idx] = copy.deepcopy(error)
                local_losses.append(loss)
            print("{}:{:.4f}".format(idx, loss), end=" ")
        print("本轮设备总用时：{:.4f}".format(time.time() - start))
        print()

        return num_current, avg_iter, layers_diff_epochs

    def train(self):
        # 记录日志和结果
        log_path = os.path.join(self.result_dir, "iid" if self.iid else "noniid", "log", "log.txt")
        result_path = os.path.join(self.result_dir, "iid" if self.iid else "noniid")

        # 加载模型
        global_model = self.load_model()
        print(global_model)
        inspect_model_keys(global_model)

        # load dataset and user groups
        train_dataset, test_dataset, user_groups = self.init_data()

        self.print_info(user_groups)
        self.record_base_message(log_path)

        # # 初始化 HyperNetwork 控制器
        # print(">>> Initializing HyperNetwork Controller...")
        # # 注意：这里只传 device，具体的功率列表我们在 train 循环里动态生成
        # self.hn_controller = HyperNetController(self.global_model, self.device)
        #
        # # 准备服务器代理数据 (Proxy Data)
        # # 从测试集随机取一个 Batch (例如 64 张图) 用于 HyperNet 的反向传播训练
        # # 假设你有 self.test_dataset，这里模拟一下
        # # 实际代码: self.proxy_data_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=True)
        # # self.proxy_batch = next(iter(self.proxy_data_loader))
        # self.proxy_data = torch.randn(64, 3, 32, 32).to(self.device)
        # self.proxy_target = torch.randint(0, 10, (64,)).to(self.device)

        # up_power_list, down_power_list, up_rate_list, down_rate_list, energy_flops_list
        if self.args.B == 20000000:
            up_power_list = [0.039] * self.args.num_users
            down_power_list = [0.135] * self.args.num_users
            up_rate_list = [7.20074143e+05, 8.50526196e+04, 1.27390715e+05, 6.27862237e+05, 8.16232570e+03, 1.68814886e+06, 1.67114339e+07, 5.96786215e+03, 1.99157937e+04, 7.46010709e+04]
            down_rate_list = [1.83466795e+06, 3.30115329e+05, 4.23763873e+05, 1.60756144e+06, 3.10756015e+04, 3.58172508e+06, 2.04733830e+07, 1.97518545e+04, 7.04373282e+04, 2.33684844e+05]

        elif self.args.B == 40000000:
            up_power_list = [0.177] * self.args.num_users
            down_power_list = [0.299] * self.args.num_users
            up_rate_list = [2.78906368e+06, 3.66580030e+05, 5.44552442e+05, 2.46516265e+06, 3.57284958e+04, 5.82329559e+06, 3.79430803e+07, 2.61345909e+04, 8.69665505e+04, 3.22197852e+05]
            down_rate_list = [3953841.74338643, 726852.06672074, 931574.4303765, 3473989.47385404, 68787.06693383, 7588129.09026439, 41535046.34088354, 43730.66202486, 155803.11917975, 515389.31248804]
        elif self.args.B == 80000000:
            up_power_list = [0.21] * self.args.num_users
            down_power_list = [0.439] * self.args.num_users
            up_rate_list = [3.73054662e+06, 4.55677370e+05, 6.80835787e+05, 3.26740777e+06, 4.39295001e+04, 8.39896421e+06, 7.02674188e+07, 3.21231523e+04, 1.07111275e+05, 3.99926372e+05]
            down_rate_list = [6.27547415e+06, 1.08469886e+06, 1.39636742e+06, 5.46922643e+06, 1.01154804e+05, 1.26799514e+07, 7.95058247e+07, 6.42711647e+04, 2.29572008e+05, 7.65567065e+05]

        energy_flops_list = [3.71099821e-11, 3.92762819e-11, 3.70683655e-11, 5.67474343e-11, 3.36262083e-11, 7.71930113e-11, 4.41594905e-11, 5.51192665e-11, 5.33593707e-11, 3.36406467e-11]

        # copy weights
        global_weights = global_model.state_dict()
        record_aggerated_model = global_model.state_dict()

        # Training
        train_losses = []
        test_accs = []

        record_js_divergences = []
        self.args.priority_list = []
        record_priority_list = []

        # hybrid_priority_list = list()
        if self.args.model == "cnn":
            hybrid_priority_list = ['classifier.2', 'features.0']
        elif self.args.model == "vgg":
            hybrid_priority_list = ['classifier.4', 'features.0']
        elif self.args.model == 'resnet':
            hybrid_priority_list = ['linear', 'conv1']

        if self.args.active_hybrid_down:
            record_hybrid_priority_list = list()

        all_train_data = np.array([])
        # 储存每台设备训练过的参数
        all_train_weights = dict()
        # 储存每台设备的error
        all_train_error = dict()
        # 存储每台设备的mask_is_frozen
        all_mask_is_frozen = dict()





        for k, v in user_groups.items():
            all_train_data = np.concatenate((all_train_data, v), axis=0)
            all_train_weights[k] = []
            all_train_error[k] = get_model_like_tensor(global_model)
            # all_mask_is_frozen[k] = get_model_like_tensor(global_model)


        # print(all_train_data)
        print(all_train_weights)



        self.reset_seed() #

        # 第一次评估
        # loss = self.get_loss(all_train_data, train_dataset, global_model)
        # Test inference after completion of training
        test_acc, test_loss = test_inference(global_model, test_dataset, self.device)   # batchsize为128吗
        test_accs.append(test_acc)
        train_losses.append(test_loss)

        initial_divergence = initialize_param_tracker(global_model)
        record_js_divergences.append(initial_divergence)

        # 记录全局模型每轮的稳定性比例
        stable_pcts_per_round = [0]
        # 记录冻结参数
        frozen_pcts = [0]
        print("-train loss:{:.4f} -test acc:{:.4f}".format(test_loss, test_acc))

        # 记录全局模型的模型每层之间的差异
        global_layers_diff = []



        mask = get_model_like_tensor(global_model)


        # # 初始化error
        # error = get_model_like_tensor(global_model)

        # 根据模型设置上传速度MB/s和训练时间s.只包括总的时间为10的状态


        for epoch in range(self.args.epochs):
            start = time.time()
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch} |\n')


            # 选择设备，并进行训练
            global_model.train()
            idxs_users = np.random.choice(range(self.args.num_users), self.m, replace=False)    # 随机选择10个用户机
            num_current, avg_iter, layers_diff_epochs = \
                self.client_train(idxs_users, global_model, user_groups, epoch, train_dataset,  # 会更新local-weights和local-losses
                                  train_losses, local_weights, local_losses, mask, all_train_weights, all_train_error)
            global_layers_diff.append(layers_diff_epochs)
            print(layers_diff_epochs)




            # 无效轮
            if len(local_weights) == 0:
                train_losses.append(train_losses[-1])
                test_accs.append(test_accs[-1])
                continue

            # get averaged weight
            pre_model = copy.deepcopy(global_model)
            record_model = copy.deepcopy(global_model.state_dict())

            # # 探针 1: 检查 "record_model"
            # # 预期: 包含所有键 (包括 Parameters 和 Buffers)
            # print_state_dict_keys(record_model, title="1. 检查 'record_model' (完整的全局模型)")
            #
            # # 假设 local_weights 是从客户端返回的
            # # [ (num_samples_1, state_dict_1), (num_samples_2, state_dict_2), ... ]
            # # 您可能也想检查一下 local_weights[0][1]
            # print_state_dict_keys(local_weights, title="检查 'local_weights[0][1]' (来自客户端)")
            #
            # # 探针 2: 检查 "average_weights_new" 的输出
            # # 怀疑点: 如果客户端没有发送缓冲区, avg_w 从这里开始就会丢失键
            # avg_w = average_weights_new(local_weights)
            # print_state_dict_keys(avg_w, title="2. 检查 'average_weights_new' 的输出 (avg_w)")
            #
            #
            # # 探针 3: 检查 "scale_weights" 的输出
            # # (这一步通常是安全的, 但以防万一)
            # scaled_avg_w = scale_weights(avg_w, self.args.global_lr)
            # print_state_dict_keys(scaled_avg_w, title="3. 检查 'scale_weights' 的输出 (scaled_avg_w)")
            #
            #
            # # 探针 4: 检查 "calculate_sum" 的输出
            # # *** 重大怀疑点: 您的 "intersection" 逻辑在这里 ***
            # global_weights = calculate_sum(record_model, scaled_avg_w)
            # print_state_dict_keys(global_weights, title="4. 检查 'calculate_sum' 的输出 (最终的 global_weights)")

            # global_weights = average_weights(local_weights)     # local_weights中的元素是[len(user_groups[idx]), copy.deepcopy(w)]的形式 是对所有数据平均吗?
            global_weights = calculate_sum(record_model, scale_weights(average_weights_new(local_weights), self.args.global_lr))

            # divergence
            global_model.load_state_dict(global_weights)
            if epoch > 0:
                record_js_divergence = calculate_js_divergence_for_models(pre_model, global_model)
                print(f"The JS Divergence in Epoch {epoch} is {record_js_divergence}", epoch, record_js_divergence)
                record_js_divergences.append(record_js_divergence)
                self.args.priority_list = _get_block_priority(record_js_divergence)
                print(f">>> Priority List in Epoch {epoch}: {self.args.priority_list}", epoch, self.args.priority_list)
                record_priority_list.append(self.args.priority_list)

            if epoch > 0 and self.args.active_hybrid_down:
                hybrid_priority_list = self.args.priority_list[:self.args.number_of_layer]
                record_hybrid_priority_list.append(hybrid_priority_list)



            # record_global_weights = copy.deepcopy(global_weights)   # 记录本轮聚合的模型


            if self.args.is_FedAvg == False and self.args.is_uptopk == False and epoch > 0:
                if self.args.layer_wise_down == True and (epoch + 1) <= self.args.start_epoch_layer_down:

                    # print("\ndownload_from_last_layer", end=" ")
                    # topK_layerwise(record_aggerated_model, global_weights, self.args.down_ratio, mask)

                    # static
                    # print("\nlayer_wise_download", end=" ")
                    # topK_layerwise_new(record_aggerated_model, global_weights, self.args.down_ratio, mask, self.args.priority_list)

                    # adptive
                    # print("\nadaptive_layer_wise_download", end=" ")
                    # topK_layerwise_new(record_aggerated_model, global_weights, self.args.down_ratio, mask,
                    #                    self.args.priority_list)

                    #hybrid
                    print("\nhybrid_layer_wise_download")
                    print(hybrid_priority_list)
                    topK_custom_hybrid(record_aggerated_model, global_weights, self.args.down_ratio, mask,
                                       hybrid_priority_list)

                else:
                    topK(record_aggerated_model, global_weights, 1 - self.args.down_ratio, mask)
                recover_model_from_mask(global_weights, record_aggerated_model, mask)




            # record_aggerated_model = copy.deepcopy(record_global_weights)
            record_aggerated_model = copy.deepcopy(global_weights)

            # update global weight
            # pre_model = copy.deepcopy(global_model.state_dict())
            replace_values(record_model, global_weights)
            global_model.load_state_dict(record_model)

            # global_model.load_state_dict(global_weights)

            # loss = self.get_loss(all_train_data, train_dataset, global_model)
            # Test inference after completion of training
            test_acc, test_loss = test_inference(global_model, test_dataset, self.device)
            if test_loss < train_losses[0] * 100: #如何比较的？
                test_accs.append(test_acc)
                train_losses.append(test_loss)

            else:
                print("recover model test_loss/test_acc : {}/{}".format(test_loss, test_acc))
                train_losses.append(train_losses[-1])
                test_accs.append(test_accs[-1])
                global_model.load_state_dict(pre_model)






            if (epoch + 11) % 10 == 0 or epoch == self.args.epochs - 1:
                save_result(self.args, os.path.join(result_path, "train_loss.txt"),
                            str(train_losses)[1:-1])
                save_result(self.args, os.path.join(result_path, "test_accuracy.txt"),
                            str(test_accs)[1:-1])
                save_result(self.args, os.path.join(result_path, "js_divergence.txt"),
                            str(record_js_divergences)[1:-1])
                save_result(self.args, os.path.join(result_path, "priority_list.txt"),
                            str(record_priority_list)[1:-1])
                if self.args.active_hybrid_down:
                    save_result(self.args, os.path.join(result_path, "hybrid_priority_list.txt"),
                                str(record_hybrid_priority_list)[1:-1])


            print("epoch{:4d} - loss: {:.4f} - accuracy: {:.4f} - lr: {:.4f} - time: {:.2f}".
                  format(epoch, test_loss, test_acc, self.args.lr, time.time() - start))
            print()


if __name__ == "__main__":
    # print(torch.__version__)
    args = args_parser()
    args.model = "cnn"

    args.upload_iter = 50
    args.upload_stable_params = 0  # 1 最后上传所有参数（不做改变）
    args.double_check = 0  # 1 使用双检测

    args.global_mask_frozen = 0  # 1 使用全局冻结
    args.result_dir = "test"

    # args.local_iter = None
    # print(torch.__version__)
    t = CentralTraining(args, iid=False, unequal=False, result_dir=args.result_dir)
    t.train()
