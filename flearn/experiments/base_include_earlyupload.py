import torch
import numpy as np
import random

from flearn.utils.model_util import test_inference, average_weights, get_model_like_tensor, update_model_stability, \
    calc_num_stable_params, update_model_stability_ignore_frozen, eval_avg_P_layers, update_model_stability_ignore_frozen_by_ratio, \
    average_weights_new, scale_weights, calculate_sum, replace_values, topK, recover_model_from_mask
from flearn.utils.update import LocalUpdate
from flearn.utils.options import args_parser
from flearn.utils.util import record_log, save_result
from flearn.models import cnn, vgg, resnet18, resnet, resnet9
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
            global_model = resnet9.ResNet9(num_classes=num_classes)
        else:
            exit('Error: unrecognized model')

        self.load_model_info(global_model)  # 无

        # Set the model to train and send it to device.
        global_model.to(self.device)
        global_model.train()    # 设置成train模式
        return global_model

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
                     local_losses, local_stable_pcts_final, local_stable_pcts_upload, local_stable_pcts_dcheck, mask, all_train_weights, all_train_error, all_speed_comms, all_train_times):
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
                model=copy.deepcopy(global_model), global_round=epoch, speed_comm=all_speed_comms[idx], t_train=all_train_times[idx],
                layers_diff_epochs=layers_diff_epochs, global_mask=mask, error=all_train_error[idx], all_train_weights=all_train_weights[idx])

            # w, loss, stable_pct_final, stable_pct_upload, stable_pct_dcheck, error = local_model.update_weights(
            #     model=copy.deepcopy(global_model), global_round=epoch, speed_comm=all_speed_comms[idx], t_train=all_train_times[idx],
            #     layers_diff_epochs=layers_diff_epochs, global_mask=mask, error=all_train_error[idx], all_train_weights=all_train_weights[idx])

            # if len(all_train_weights[idx]) == 0 or len(all_train_weights[idx]) == 1:
            #     w, loss, stable_pct_final, stable_pct_upload, stable_pct_dcheck, error = local_model.update_weights(
            #         model=copy.deepcopy(global_model), global_round=epoch,
            #         layers_diff_epochs=layers_diff_epochs, global_mask=mask, error=all_train_error[idx])
            #
            #     if loss < train_losses[0] * 100:
            #         all_train_weights[idx].append(copy.deepcopy(global_model.state_dict()))
            #
            # else:
            #
            #     if len(all_train_weights[idx]) == 2:
            #         new_model = local_model.update_model(global_model, all_train_weights[idx])
            #         g_model.load_state_dict(new_model)
            #         w, loss, stable_pct_final, stable_pct_upload, stable_pct_dcheck, error = local_model.update_weights(
            #             model=copy.deepcopy(g_model), global_round=epoch,
            #             layers_diff_epochs=layers_diff_epochs, global_mask=mask, error=all_train_error[idx])
            #
            #         if loss < train_losses[0] * 100:
            #             all_train_weights[idx][-2] = copy.deepcopy(all_train_weights[idx][-1])
            #             all_train_weights[idx][-1] = copy.deepcopy(new_model)




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
            # local_stable_pcts_final.append(stable_pct_final)
            # local_stable_pcts_upload.append(stable_pct_upload)
            # local_stable_pcts_dcheck.append(stable_pct_dcheck)
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

        # load dataset and user groups
        train_dataset, test_dataset, user_groups = self.init_data()

        self.print_info(user_groups)
        self.record_base_message(log_path) #会重写logpath吗

        # copy weights
        global_weights = global_model.state_dict()

        # Training
        train_losses = []
        test_accs = []

        all_train_data = np.array([])
        # 储存每台设备训练过的参数
        all_train_weights = dict()
        # 储存每台设备的error
        all_train_error = dict()
        # 存储每台设备的mask_is_frozen
        all_mask_is_frozen = dict()
        if self.args.model == "cnn":
            all_train_times = [0.1] * self.args.num_users
            all_speed_comms = [10] * self.args.num_users
        elif self.args.model == "vgg":
            all_train_times = [1] * self.args.num_users
            all_speed_comms = [300] * self.args.num_users
        elif self.args.model == "resnet":
            all_train_times = [1] * self.args.num_users
            all_speed_comms = [300] * self.args.num_users

        # if self.args.num_users == 10:
        #     if self.args.model == "cnn":
        #         speed_comms = [0.392, 0.017, 0.128, 0.015, 0.046, 0.03, 0.016, 0.021, 0.016, 0.015]
        #         train_times = [10.442, 6.554, 10.077, 5.906, 9.114, 8.314, 6.296, 7.328, 6.369, 5.883]
        #     elif self.args.model == "vgg":
        #         speed_comms = [1.031, 0.047, 0.349, 0.041, 0.127, 0.083, 0.044, 0.058, 0.045, 0.04]
        #         train_times = [351.902, 220.891, 339.617, 199.031, 307.157, 280.194, 212.169, 246.965, 214.644, 198.281]
        #     elif self.args.model == "resnet":
        #         speed_comms = [1.031, 0.047, 0.349, 0.041, 0.127, 0.083, 0.044, 0.058, 0.045, 0.04]
        #         train_times = [351.902, 220.891, 339.617, 199.031, 307.157, 280.194, 212.169, 246.965, 214.644, 198.281]

        # if self.args.num_users == 10:
        #     if self.args.model == "cnn":
        #         speed_comms = [0.397, 0.035, 0.21, 0.031, 0.075, 0.06, 0.035, 0.038, 0.032, 0.03]
        #         train_times = [11.486, 7.21, 11.085, 6.496, 10.026, 9.145, 6.925, 8.061, 7.006, 6.472]
        #     elif self.args.model == "vgg":
        #         speed_comms = [0.539, 0.041, 0.251, 0.035, 0.104, 0.07, 0.038, 0.05, 0.039, 0.035]
        #         train_times = [387.092, 242.98, 373.579, 218.935, 337.873, 308.213, 233.386, 271.662, 236.108, 218.109]
        #     elif self.args.model == "resnet":
        #         # speed_comms = [0.413, 0.037, 0.211, 0.032, 0.092, 0.063, 0.035, 0.045, 0.035, 0.032]
        #         speed_comms = [0.734, 0.021, 0.175, 0.018, 0.058, 0.037, 0.02, 0.026, 0.02, 0.018]
        #         train_times = [957.698, 601.152, 924.266, 541.663, 835.927, 762.546, 577.416, 672.114, 584.151, 539.621]

        if self.args.num_users == 10:
            if self.args.model == "cnn":
                # speed_comms = [0.464, 0.07, 0.415, 0.066, 0.113, 0.113, 0.073, 0.064, 0.063, 0.062]     # SCA
                # speed_comms = [0.801, 0.426, 0.457, 0.795, 0.099, 0.977, 1.675, 0.086, 0.198, 0.344]    # B/N
                speed_comms = [1.409, 0.239, 0.408, 0.658, 0.11, 0.521, 0.799, 0.044, 0.175, 0.238]     # B_random
                train_times = [10.618, 5.064, 9.952, 4.387, 8.365, 7.208, 4.788, 5.946, 4.865, 4.365]
            elif self.args.model == "vgg":
                # speed_comms = [1.093, 0.105, 0.515, 0.095, 0.228, 0.162, 0.101, 0.123, 0.102, 0.095]    # SCA
                # speed_comms = [1.199, 0.686, 0.734, 1.184, 0.181, 1.437, 2.48, 0.156, 0.345, 0.573]    # B/N
                speed_comms = [1.355, 1.251, 0.351, 0.249, 0.102, 2.005, 1.426, 0.138, 0.316, 0.612]    # B_random
                train_times = [357.836, 170.678, 335.405, 147.863, 281.91, 242.903, 161.359, 200.403, 163.973, 147.111]
            elif self.args.model == "resnet":
                # speed_comms = [0.413, 0.037, 0.211, 0.032, 0.092, 0.063, 0.035, 0.045, 0.035, 0.032]
                # speed_comms = [3.472, 0.027, 0.215, 0.024, 0.066, 0.044, 0.026, 0.032, 0.026, 0.024]    # SCA
                # speed_comms = [1.056, 0.439, 0.496, 1.028, 0.064, 1.396, 3.663, 0.052, 0.154, 0.338]     # B/N
                speed_comms = [2.045, 0.244, 0.44, 0.833, 0.068, 0.696, 1.599, 0.032, 0.139, 0.235]     # B_random

                train_times = [885.316, 422.272, 829.821, 365.827, 697.47, 600.964, 399.217, 495.814, 405.684, 363.966]

            for k, v in user_groups.items():
                all_train_times[k] = train_times[k]
                all_speed_comms[k] = speed_comms[k]

        for k, v in user_groups.items():
            all_train_data = np.concatenate((all_train_data, v), axis=0)
            all_train_weights[k] = []
            all_train_error[k] = get_model_like_tensor(global_model)
            # all_mask_is_frozen[k] = get_model_like_tensor(global_model)
        print(all_train_times)
        print(all_speed_comms)

        # print(all_train_data)
        print(all_train_weights)



        self.reset_seed() #

        # 第一次评估
        # loss = self.get_loss(all_train_data, train_dataset, global_model)
        # Test inference after completion of training
        test_acc, test_loss = test_inference(global_model, test_dataset, self.device)   # batchsize为128吗
        test_accs.append(test_acc)
        train_losses.append(test_loss)
        # 记录最终稳定性比例
        stable_pcts_final = [0]
        # 记录上传时稳定性比例
        stable_pcts_upload = [0]
        # 记录上传和最终重合的比例
        stable_pcts_dcheck = [0]
        # 记录全局模型每轮的稳定性比例
        stable_pcts_per_round = [0]
        # 记录冻结参数
        frozen_pcts = [0]
        print("-train loss:{:.4f} -test acc:{:.4f}".format(test_loss, test_acc))

        # 记录全局模型的模型每层之间的差异
        global_layers_diff = []


        # 稳定性
        alpha = 0.99
        threshold = 0.05 #?
        ratio = 0.2
        E = get_model_like_tensor(global_model)
        E_abs = get_model_like_tensor(global_model)
        mask = get_model_like_tensor(global_model)
        L = get_model_like_tensor(global_model)
        I = get_model_like_tensor(global_model)

        # # 初始化error
        # error = get_model_like_tensor(global_model)

        check_interval = 10
        last_w = copy.deepcopy(global_model.state_dict())
        Ps_layers = [] #?

        # 根据模型设置上传速度MB/s和训练时间s.只包括总的时间为10的状态


        for epoch in range(self.args.epochs):
            start = time.time()
            local_weights, local_losses = [], []
            local_stable_pcts_final = []
            local_stable_pcts_upload = []
            local_stable_pcts_dcheck = []
            print(f'\n | Global Training Round : {epoch} |\n')

            # 选择设备，并进行训练
            global_model.train()
            idxs_users = np.random.choice(range(self.args.num_users), self.m, replace=False)    # 随机选择10个用户机
            num_current, avg_iter, layers_diff_epochs = \
                self.client_train(idxs_users, global_model, user_groups, epoch, train_dataset,  # 会更新local-weights和local-losses
                                  train_losses, local_weights, local_losses, local_stable_pcts_final,
                                  local_stable_pcts_upload, local_stable_pcts_dcheck, mask, all_train_weights, all_train_error, all_speed_comms, all_train_times)
            global_layers_diff.append(layers_diff_epochs)
            print(layers_diff_epochs)




            # 无效轮
            if len(local_weights) == 0:
                train_losses.append(train_losses[-1])
                test_accs.append(test_accs[-1])
                stable_pcts_final.append(stable_pcts_final[-1])
                stable_pcts_upload.append(stable_pcts_upload[-1])
                stable_pcts_dcheck.append(stable_pcts_dcheck[-1])
                continue

            # get averaged weight
            pre_model = copy.deepcopy(global_model.state_dict())
            record_model = copy.deepcopy(global_model.state_dict())
            # global_weights = average_weights(local_weights)     # local_weights中的元素是[len(user_groups[idx]), copy.deepcopy(w)]的形式 是对所有数据平均吗?
            global_weights = calculate_sum(pre_model, scale_weights(average_weights_new(local_weights), self.args.global_lr))
            # global_weights = calculate_sum(pre_model, scale_weights(average_weights_new(local_weights), 900/(1000 + epoch)))
            # global_weights = average_weights_new(local_weights)

            record_global_weights = copy.deepcopy(global_weights)   # 记录本轮聚合的模型


            if self.args.is_FedAvg == False and self.args.is_uptopk == False and epoch > 0:
                topK(record_aggerated_model, global_weights, 1 - self.args.down_ratio, mask)
                recover_model_from_mask(global_weights, record_aggerated_model, mask)




            record_aggerated_model = copy.deepcopy(record_global_weights)

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
                stable_pcts_final.append(round(sum(local_stable_pcts_final) / len(local_stable_pcts_final), 4))
                stable_pcts_upload.append(round(sum(local_stable_pcts_upload) / len(local_stable_pcts_upload), 4)) #这个比值
                stable_pcts_dcheck.append(round(sum(local_stable_pcts_dcheck) / len(local_stable_pcts_dcheck), 4))
            else:
                print("recover model test_loss/test_acc : {}/{}".format(test_loss, test_acc))
                train_losses.append(train_losses[-1])
                test_accs.append(test_accs[-1])
                stable_pcts_final.append(stable_pcts_final[-1])
                stable_pcts_upload.append(stable_pcts_upload[-1])
                stable_pcts_dcheck.append(stable_pcts_dcheck[-1])
                global_model.load_state_dict(pre_model)

            # 进行稳定性检查， 每次检查完毕后，就要变化上一个模型参数 ？ last_w和pre_model
            # if (epoch + 1) % check_interval == 0:
            #     # update_model_stability_ignore_frozen(last_w, global_model.state_dict(), E, E_abs, mask,
            #     #                                      L, I, epoch, check_interval, alpha, threshold)
            #     update_model_stability_ignore_frozen_by_ratio(last_w, global_model.state_dict(), E, E_abs, mask,
            #                                          L, I, epoch, check_interval, alpha, ratio)
            #     last_w = copy.deepcopy(global_model.state_dict())

            # 统计该轮的稳定参数量
            stable_pct_per_round = calc_num_stable_params(mask)
            stable_pcts_per_round.append(stable_pct_per_round)

            # 统计每层平均稳定参数指数
            Ps_layers.append(eval_avg_P_layers(E, E_abs))

            if (epoch + 11) % 10 == 0 or epoch == self.args.epochs - 1:
                save_result(self.args, os.path.join(result_path, "train_loss.txt"),
                            str(train_losses)[1:-1])
                save_result(self.args, os.path.join(result_path, "test_accuracy.txt"),
                            str(test_accs)[1:-1])
                # save_result(self.args, os.path.join(result_path, "diff_model.txt"),
                #             str(global_layers_diff)[1:-1])
                save_result(self.args, os.path.join(result_path, "stable_pct_final.txt"),
                            str(stable_pcts_final)[1:-1])
                save_result(self.args, os.path.join(result_path, "stable_pct_upload.txt"),
                            str(stable_pcts_upload)[1:-1])
                save_result(self.args, os.path.join(result_path, "stable_pct_dcheck.txt"),
                            str(stable_pcts_dcheck)[1:-1])
                save_result(self.args, os.path.join(result_path, "stable_pcts_per_round.txt"),
                            str(stable_pcts_per_round)[1:-1])
                save_result(self.args, os.path.join(result_path, "Ps_layers_per_round.txt"),
                            str(Ps_layers)[1:-1])

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
