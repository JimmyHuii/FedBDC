import wandb
import tensorflow as tf

import datetime

import math

import random
from copy import deepcopy

import simpy
import Utils
from database import Database


from model import model_test as test_cnn

from model import amd_train_cnn as train_cnn
from model import amd_grad

from options import args_parser
import numpy as np
import time


from RL_selector import RL_selector


# 需要在updater和scheduler之间传递的变量定义为全局变量。

w_global = []
k_global = 0
gradients_count = 0
training_devices = set()

dropped_device_count = 0
aggregated_device_count = 0

w_amd = []  # 最新的额外下发模型
k_amd = 0   # w_amd的epoch
t_amd = 0   # 下发w_amd的时间
can_amd = True

thetas = []
gammas = []

lambdas = []
bs = []
a_lst = []
dw_dlambda = []
dw_db = []
dw_da = []

Q_lst = []
R_lst = []
status_lst = []


init_Rl = 0

rl_reward_mean_lst = []

single_grad_latency_lst = []


#from network.basic import ResNet20, CNN_SVHN, LeNet, TFCNN
from get_model import get_model, get_optimizer, get_loss_fn

#input_shape = (32, 32, 3)
#num_classes = 100
local_model = 0
global_model = 0
optimizer = 0
loss_fn = 0

can_trigger = True
def worker(env, device_id, train, dataset, dataset_lst, device_latency_lst, k_down, local_weights_store, worker_info_lst, args):
    global gammas
    global thetas
    global Q_lst
    global R_lst
    global status_lst
    global training_devices
    global init_Rl
    global rl_reward_mean_lst
    global single_grad_latency_lst
    global local_model
    global optimizer
    global loss_fn
    device_w0 = deepcopy(w_global)  # 在此时获取全局模型，时机很重要
    local_weights = deepcopy(device_w0)
    local_lr = args.local_lr * \
        (args.local_lr_decay_rate ** (k_down // args.decay_interval))
    local_lr = max(local_lr, args.local_lr_min)
    print(f"local lr: {local_lr}")
    amd_dic = {'amd': False}
    k_down_new = "#"

    start = env.now
    # print(f"device {device['id']} 开始 下发 {env.now}")
    yield env.timeout(device_latency_lst[device_id][0])
    # print(f"device {device['id']} 结束 下发 {env.now}")
    down_load_end = env.now
    # print(f"device {device['id']} 开始 训练 {env.now}")

    ###################################################################
    sampled_data_size = math.ceil(
        len(dataset_lst[device_id])/args.local_dataset_sample)
    train_data = dataset.sample_train_data_by_index(random.sample(
        dataset_lst[device_id], sampled_data_size))

    (x_train, y_train) = train_data     # 将采样得到的训练数据拆分成输入特征 x_train 和标签 y_train
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(
        buffer_size=1024)
    
    # 计算剩余的样本数
    remainder = len(x_train) % args.batch_size
    if remainder > 0:
        # 计算需要的额外样本数
        extra_samples = args.batch_size - remainder
        # 对训练数据进行过采样以获取额外的样本
        extra_indices = np.random.choice(len(x_train), extra_samples)
        extra_x_train = tf.gather(x_train, extra_indices)
        extra_y_train = tf.gather(y_train, extra_indices)
        # 添加额外的样本到训练数据集
        train_dataset = train_dataset.concatenate(tf.data.Dataset.from_tensor_slices((extra_x_train, extra_y_train)))

    # 最后对数据进行分批
    train_dataset = train_dataset.batch(args.batch_size)

    # print("train_dataset shape: ")
    # print([x.shape for x, y in train_dataset])
    

    #train_latency = device_latency_lst[device_id][1]
    bias = 0
    local_grads = args.local_epochs * \
        math.ceil(sampled_data_size/args.batch_size)
    one_iteration_latency = single_grad_latency_lst[device_id]
    train_latency = one_iteration_latency * local_grads

    #status = status_lst[device_id]
    local_data_size = len(dataset_lst[device_id])
    down_latency = device_latency_lst[device_id][0]
    upload_latency = device_latency_lst[device_id][2]
    batch_size = args.batch_size
    first_selected = False
    if status_lst[device_id] == -1:
        first_selected = True
        init_status = math.floor(init_Rl.predict(
            local_data_size, batch_size, down_latency, train_latency, upload_latency)*local_grads)
        status_lst[device_id] = init_status
        Q_lst[device_id][init_status] = [0] * 3
    # else:
    #    status = status_lst[device_id]
    status = status_lst[device_id]

    # status: request 时机
    # Action:
    # 0: 前移， 1：不变,  2:后移
    # 0: status -= 1， 1：status 不变， 2：status += 1
    if status == 0:
        action = Q_lst[device_id][status][1:].index(
            max(Q_lst[device_id][status][1:])) + 1
        if random.random() < args.rl_epsilon:
            action = random.randint(1, 2)
    elif status == local_grads - 2:
        action = Q_lst[device_id][status][:-
                                          1].index(max(Q_lst[device_id][status][:-1]))
        if random.random() < args.rl_epsilon:
            action = random.randint(0, 1)
    else:
        action = Q_lst[device_id][status].index(max(Q_lst[device_id][status]))
        if random.random() < args.rl_epsilon:
            action = random.randint(0, 2)
    # print(f"device {device_id} status {status} action {action}")

    if action == 0:
        status -= 1
    elif action == 2:
        status += 1

    if status not in Q_lst[device_id]:
        Q_lst[device_id][status] = [0] * 3

    l_star = status
    # print(f'设备{device_id}的l_star为{l_star}')
    agg_decay = random.randint(1, 10)
    part1_iterations = l_star + agg_decay

    if part1_iterations > 0:
        if part1_iterations > local_grads:
            part1_iterations = local_grads
        # print(f'设备{device_id}的bias为{bias}, part1_iterations为{part1_iterations}, local_grads为{local_grads}')
        #local_weights, grad_tau, last_loss = train_cnn_(
        #    local_weights, train_data, part1_iterations, train_dataset, args, device_id, bias, local_lr)
        local_model.set_weights(local_weights)
        optimizer.learning_rate = local_lr
        grad_tau = amd_grad(
            local_model, loss_fn, train_data, part1_iterations, train_dataset, args, device_id, bias)
        local_weights, last_loss = train_cnn(
            local_model, optimizer, loss_fn, train_data, part1_iterations, train_dataset, args, device_id, bias)
        bias += part1_iterations
        yield env.timeout(one_iteration_latency * part1_iterations)
    part2_iterations = local_grads - part1_iterations

    d_loss = 0
    local_agg_flag = False
    grad_tau_mid = -1
    if part2_iterations > 0:
        for j in range(part2_iterations):
            if k_amd > k_down:
                # 如果有新的额外模型下发
                print(f'Client {device_id} received fresh model k_amd {k_amd}')
                amd_dic['amd'] = True
                amd_dic['amd_time'] = env.now
                new_global_weights = deepcopy(w_amd)
                current_k_amd = k_amd
                amd_dic['from_round'] = current_k_amd
                # beta_k = args.beta * (current_k_amd - k_down + 1) ** (-args.b)
                # beta_k = args.beta

                if args.real_local_agg > 0:
                    tp = gammas[device_id] * \
                        (1 - thetas[device_id] / math.sqrt(current_k_amd -
                         k_down + 1)) / math.sqrt(current_k_amd)
                    beta_k = 1 / (1 + tp)

                    dw_dgamma = [np.array(-beta_k*beta_k*(1 - thetas[device_id] / math.sqrt(current_k_amd -
                                                                                            k_down + 1))*(np.array(local_weights[i]) - new_global_weights[i]) / math.sqrt(current_k_amd)) for i in range(len(local_weights))]
                    if abs(args.beta_gamma_lr) > 1e-9:
                        gammas[device_id] -= args.beta_gamma_lr * \
                            sum([np.sum(grad_tau[i].flatten()*dw_dgamma
                                [i].flatten()) for i in range(len(grad_tau))])

                    dw_dtheta = [np.array(beta_k*beta_k*gammas[device_id]*(np.array(local_weights[i]) - new_global_weights[i]) / math.sqrt(
                        current_k_amd - k_down + 1) / math.sqrt(current_k_amd)) for i in range(len(local_weights))]

                    if abs(args.beta_theta_lr) > 1e-9:
                        thetas[device_id] -= args.beta_theta_lr * \
                            sum([np.sum(grad_tau[i].flatten()*dw_dtheta
                                [i].flatten()) for i in range(len(grad_tau))])

                    tp = gammas[device_id] * \
                        (1 - thetas[device_id] / math.sqrt(current_k_amd -
                         k_down + 1)) / math.sqrt(current_k_amd)
                    beta_k = 1/(1+tp)
                else:
                    beta_k = 1
                # print(f"device {device_id} current_k_amd={current_k_amd}, k_down={k_down}, beta_k={beta_k}, beta_gamma={gammas[device_id]}, beta_theta={thetas[device_id]}")
                #print(args.beta, beta_k, k_down, current_k_amd)
                
                better_device_w0 = []
                for i, layer_weights in enumerate(local_weights):
                    better_device_w0.append(
                        np.array(np.array(layer_weights) * beta_k + new_global_weights[i] * (1 - beta_k)))
                # print(f'设备{device_id}的bias为{bias}, part2_iterations为{j}, local_grads为{local_grads}')

                                
                local_model.set_weights(better_device_w0)
                optimizer.learning_rate = local_lr
                

                
                grad_tau_mid = amd_grad(
                    local_model, loss_fn, train_data, 1, train_dataset, args, device_id, bias)
                

                
                local_weights, loss_value_agg = train_cnn(
                    local_model, optimizer, loss_fn, train_data, 1, train_dataset, args, device_id, bias)
                
                bias += 1
                local_agg_flag = True
                d_loss = last_loss - loss_value_agg
                

                # 因为额外的聚合，要修改这个模型的陈腐性，即k_down_new
                k_down_new = ((part1_iterations+j) * k_down +
                              (local_grads - (part1_iterations+j)) *
                              current_k_amd) / local_grads
                #k_down_new = current_k_amd - 1.0 * (part1_iterations + j) * (current_k_amd - k_down) / local_grads
                amd_dic['k_down_new'] = k_down_new

                yield env.timeout(one_iteration_latency)
                remain_iterations = local_grads - (part1_iterations + j + 1)
                if remain_iterations > 0:
                    # do the remaining iterations
                    # print(f'设备{device_id}的bias为{bias}, remain_iterations为{remain_iterations}, local_grads为{local_grads}')
                    #local_weights, last_loss = train_cnn(
                    #    local_weights, train_data, remain_iterations, train_dataset, args, device_id, bias, local_lr)
                    local_model.set_weights(local_weights)
                    optimizer.learning_rate = local_lr
                    local_weights, last_loss = train_cnn(
                        local_model, optimizer, loss_fn, train_data, remain_iterations, train_dataset, args, device_id, bias)
                    bias += remain_iterations
                    yield env.timeout(one_iteration_latency * remain_iterations)
                break

            else:
                # 没有额外模型下发的话就在原来的基础上继续训练
                # print(f'设备{device_id}的bias为{bias}, part2_iterations为{j}, local_grads为{local_grads}')
                #local_weights, last_loss = train_cnn(
                #    local_weights, train_data, 1, train_dataset, args, device_id, bias, local_lr)
                local_model.set_weights(local_weights)
                optimizer.learning_rate = local_lr
                local_weights, last_loss = train_cnn(
                    local_model, optimizer, loss_fn, train_data, 1, train_dataset, args, device_id, bias)
                bias += 1
                

            yield env.timeout(one_iteration_latency)

    if local_agg_flag == True:
        reward = d_loss
        if not first_selected:
            s0 = status_lst[device_id]
            if s0 not in R_lst[device_id]:
                R_lst[device_id][s0] = [0] * 3
            R_lst[device_id][s0][action] = reward
            Q_lst[device_id][s0][action] = Q_lst[device_id][s0][action] + args.rl_alpha * \
                (R_lst[device_id][s0][action] + args.rl_gamma *
                 max(Q_lst[device_id][status]) - Q_lst[device_id][s0][action])
        else:
            rl_reward_mean_lst[device_id].append(reward)
            reward_mean = np.mean(rl_reward_mean_lst[device_id][-10:])
            init_Rl.learn(local_data_size, batch_size, down_latency,
                          train_latency, upload_latency, reward, reward_mean)
        status_lst[device_id] = status
        # print(f"device {device_id} l_star {l_star} decay {agg_decay} reward {reward} Q_lst {Q_lst[device_id]}")

    ###################################################################

    # print(f"device {device['id']} 结束 训练 {env.now}")
    train_end = env.now
    # print(f"device {device['id']} 开始 上传 {env.now}")

    # 本地测试精度
    #test_data = dataset.sample_test_datas()
    #train_data_new = dataset.sample_train_data_by_index(random.sample(
    #    dataset_lst[device_id], math.ceil(len(dataset_lst[device_id])/args.local_dataset_sample)))
    
    #local_model.set_weights(local_weights)
    #loss, test_acc = test_cnn(local_model, test_data)
    #loss, train_acc = test_cnn(local_model, train_data)
    #loss, train_acc_new = test_cnn(local_model, train_data_new)


    #print(
    #    f"device {device_id} test_acc {test_acc} train_acc {train_acc} train_acc_new {train_acc_new}")
    # print(f"device {device_id} local_grads {local_grads} single_grad_latency {single_grad_latency_lst[device_id]}")
    # print(f"down_lantency {device_latency_lst[device_id][0]} train_latency {train_latency} upload_lantency {device_latency_lst[device_id][2]} total_latency {device_latency_lst[device_id][0]+train_latency+device_latency_lst[device_id][2]}")


    yield env.timeout(device_latency_lst[device_id][2])
    if amd_dic['amd']:
        yield local_weights_store.put((k_down, k_down_new, local_weights, grad_tau, grad_tau_mid, device_id))
    else:
        yield local_weights_store.put((k_down, k_down_new, local_weights, grad_tau, grad_tau_mid, device_id))
    training_devices.remove(device_id)
    # print(f"device {device['id']} 结束 上传 {env.now}")
    end = env.now
    worker_info = {'id': device_id, 'start': start, 'download_end': down_load_end,
                   'train_end': train_end, 'end': end, 'k_down': k_down, 'amd': amd_dic}
    # print(worker_info)
    worker_info_lst.append(worker_info)


def scheduler(env, train, dataset, dataset_lst, device_latency_lst, local_weights_store, worker_info_lst, args):
    global training_devices, can_trigger

    while env.now <= args.T:
        if can_trigger == False:
            yield env.timeout(args.trigger_interval)
            continue
        for i in range(10):
            if len(training_devices) <= args.max_concurrent:
                k_down = k_global
                device_id = random.choice(
                    list(set(range(args.N)) - training_devices))
                env.process(worker(env, device_id, train, dataset, dataset_lst,
                            device_latency_lst, k_down, local_weights_store, worker_info_lst, args))
                training_devices.add(device_id)
                print(f"{env.now} 触发了{device_id}, 正在训练的设备有{training_devices}")
            else:
                pass
                # print("无空闲设备")
        can_trigger = False
        yield env.timeout(args.trigger_interval)
    else:
        print(
            f"当前时间{env.now}， 当前轮数{k_global}，结束-{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if args.wandb > 0:
            wandb.finish()


def send_additional_model(env, download_time, w_global, k_global, t):
    # 这个函数在一段时间后将w_amd的值置为w_global
    global w_amd
    global k_amd
    global t_amd
    yield env.timeout(download_time)
    if can_amd == False:
        return
    w_amd = w_global
    k_amd = k_global
    t_amd = t


def updater(env, train, start_time, local_weights_store, dataset, dataset_lst, gradients_lst, time_lst, acc_lst, staleness_lst, server_info_lst, custom, args):
    global k_global
    global w_global
    global gradients_count
    global dropped_device_count
    global aggregated_device_count

    global lambdas
    global bs
    global a_lst
    global dw_dlambda
    global dw_db
    global dw_da

    global global_model
    global can_trigger

    global k_amd
    global can_amd

    received_devices_count = 0
    # 一直接收
    while True:
        # 接受一个本地模型权重
        k_down, k_down_new, local_weights, grad_tau, grad_tau_mid, device_id = yield local_weights_store.get()
        #print(f"device {device_id} 上传完成, 时间{env.now}, k_down {k_down}, k_down_new {k_down_new}")
        k_up = k_global
        staleness = k_up - k_down
        print(f"device {device_id} 上传完成, 时间{env.now}, k_up {k_up}, k_down {k_down}, k_down_new {k_down_new}")
        if k_down_new != "#":
            staleness = k_up - k_down_new
        if staleness > args.S:
            dropped_device_count += 1
            # print(f"陈腐性超出限制 跳过, 时间{env.now}, 当前陈腐性{staleness}, 已丢弃个数{dropped_device_count}, 已丢弃比例{dropped_device_count/(dropped_device_count+aggregated_device_count)}")
            continue
        else:
            # received_devices_count += 1
            aggregated_device_count += 1
            sampled_data_size = math.ceil(
                len(dataset_lst[device_id])/args.local_dataset_sample)
            local_grads = args.local_epochs * \
                math.ceil(sampled_data_size/args.batch_size)
            gradients_count += local_grads

            if train:
                # 更新 lambda 和 b
                if dw_dlambda[device_id] != 0 and dw_db[device_id] != 0 and dw_da[device_id] != 0:
                    if abs(args.lambda_lr) > 1e-9:
                        lambdas[device_id] -= args.lambda_lr * \
                            sum([np.sum(grad_tau[i].flatten()*dw_dlambda[device_id]
                                [i].flatten()) for i in range(len(grad_tau))])
                    if abs(args.b_lr) > 1e-9:
                        bs[device_id] -= args.b_lr * \
                            sum([np.sum(grad_tau[i].flatten()*dw_db[device_id]
                                [i].flatten()) for i in range(len(grad_tau))])
                    if abs(args.a_lr) > 1e-9:
                        a_lst[device_id] -= args.a_lr * \
                            sum([np.sum(grad_tau[i].flatten()*dw_da[device_id]
                                [i].flatten()) for i in range(len(grad_tau))])

                    # print(lambdas)
                    # print(bs)
                    # print(a_lst)

                alpha_t = args.alpha * \
                    (lambdas[device_id] /
                     (math.sqrt(k_up+1)*math.pow(staleness+1, a_lst[device_id])) + bs[device_id])

                # 保存梯度供下一次更新 lambda 和 b 使用
                dw_dlambda[device_id] = [np.array(args.alpha *
                                                  (np.array(local_weights[i]) - w_global[i]) /
                                                  (math.sqrt(k_up+1)*math.pow(staleness+1, a_lst[device_id])*math.pow(1+alpha_t, 2))) for i in range(len(local_weights))]

                dw_db[device_id] = [np.array(args.alpha *
                                             (np.array(local_weights[i]) - w_global[i])/math.pow(1+alpha_t, 2)) for i in range(len(local_weights))]

                dw_da[device_id] = [np.array(-1*args.alpha * lambdas[device_id]*math.log(staleness+1) *
                                             (np.array(local_weights[i]) - w_global[i]) /
                                             (math.sqrt(k_up+1)*math.pow(staleness+1, a_lst[device_id])*math.pow(1+alpha_t, 2))) for i in range(len(local_weights))]

                # 异步聚合
                # print(f"device_id {device_id} alpha_t {alpha_t} , lambda {lambdas[device_id]} , b {bs[device_id]}, a {a_lst[device_id]}, staleness {staleness}")

                #alpha_t = args.alpha * (staleness + 1) ** (-args.a)
                new_weights = []
                w1 = 1 / (1 + alpha_t)  # global weight
                w2 = 1 - w1  # local weight
                for i, layer_weights in enumerate(local_weights):
                    new_weights.append(
                        np.array(np.array(layer_weights) * w2 + w_global[i] * w1))
                w_global = deepcopy(new_weights)

                # 聚合的设备
                k_global += 1
                received_devices_count += 1
                
                if received_devices_count == 1:
                    can_amd = True
                # 判断是否要额外下发
                download_time = args.lambd / \
                    (args.lambd * args.theta + args.lambd +
                    args.theta + 1) * args.tik_min
                if env.now - t_amd >= download_time and k_global > k_amd:
                    # 如果当前没有正在下发，而且新聚合的比较新的话
                    env.process(send_additional_model(
                        env, download_time, deepcopy(w_global), k_global, env.now))

                if received_devices_count == 10:
                    #can_trigger = True
                    received_devices_count = 0
                    # k_global += 1
                    k_global -= 9
                    can_amd = False
                    k_amd = k_global
                    print(f'server k_amd {k_amd}')

                    # 测试
                    #index = k_global % args.N
                    #sample_indexes = dataset_lst[index]["test"]
                    #test_data = dataset.sample_test_datas(sample_indexes)
                    test_data = dataset.sample_test_datas()
                    global_model.set_weights(w_global)
                    loss, acc = test_cnn(global_model, test_data)

                    # 记录结果
                    gradients_lst.append(gradients_count)
                    time_lst.append(env.now)
                    acc_lst.append(acc)
                    staleness_lst.append(staleness)
                    # if k_global % args.save_interval == 0:
                    #    dic = {"start_time": start_time, 'fraction': args.fraction, 'regularization': args.regularization,
                    #           'gradients': gradients_lst, 'time_lst': time_lst, 'acc_lst': acc_lst, 'staleness_lst': staleness_lst}
                    #    Utils.save_results(dic, custom, args)

                    now_time = env.now

                    print(f"round {k_global:5d}, gradients {gradients_count:8.0f}, time {now_time:8.2f} staleness {staleness:4.2f} acc {acc:8.2%} \n 丢弃比例：{dropped_device_count/(dropped_device_count+aggregated_device_count)}")

                    if args.wandb > 0:
                        metrics = {"round":k_global, "gradients":gradients_count, "time":now_time, "acc":acc, "staleness":staleness, "drop_ratio":dropped_device_count/(dropped_device_count+aggregated_device_count)}
                        wandb.log(metrics)
                    can_trigger = True

            else:
                print(
                    f'round {k_global}, time {env.now}, gradients_count {gradients_count}, staleness {staleness}')
                pass


            #server_info = {'round': k_global, 'gradients_count': gradients_count, 'end': env.now,
                           # 'staleness': staleness, "drop_ratio": dropped_device_count/(dropped_device_count+aggregated_device_count)}
            #server_info_lst.append(server_info)

            # 判断是否要额外下发
            #download_time = args.lambd / \
            #    (args.lambd * args.theta + args.lambd +
            #     args.theta + 1) * args.tik_min
            #if env.now - t_amd >= download_time and k_global > k_amd:
            #    # 如果当前没有正在下发，而且新聚合的比较新的话
            #    env.process(send_additional_model(
            #        env, download_time, deepcopy(w_global), k_global, env.now))


def run(train, worker_info_lst, server_info_lst, custom, args):
    global k_global
    global w_global
    global gradients_count

    global dropped_device_count
    global aggregated_device_count

    global lambdas
    global bs
    global a_lst
    global gammas
    global thetas
    global dw_dlambda
    global dw_db
    global dw_da

    global Q_lst
    global R_lst
    global status_lst

    global init_Rl
    global rl_reward_mean_lst
    global single_grad_latency_lst

    global global_model
    global local_model

    global_model = get_model(args)
    local_model = get_model(args)

    global optimizer
    # 选择优化器
    optimizer = get_optimizer(args)

    global loss_fn
    # 选择损失函数
    loss_fn = get_loss_fn(args)

    # 每次训练之前要把全局变量置为初始值，要不然连着两次调用此函数会受到影响
    k_global = 0
    w_global = []
    gradients_count = 0

    dropped_device_count = 0
    aggregated_device_count = 0

    lambdas = [args.s_lambda] * args.N
    bs = [args.s_b] * args.N
    a_lst = [args.s_a] * args.N

    thetas = [args.beta_theta] * args.N
    gammas = [args.beta_gamma] * args.N

    dw_dlambda = [0] * args.N
    dw_db = [0] * args.N
    dw_da = [0] * args.N

    rl_reward_mean_lst = [[] for i in range(args.N)]

    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    env = simpy.Environment()   # 创建了一个 SimPy 环境对象 env，该对象用于模拟并驱动联邦学习的训练过程。

    w0 = Utils.load_initial_variables(args, args.initial_weights_filename)
    w_global = deepcopy(w0)
    dataset = Database(args.dataset, args.seed)
    dataset_lst = Utils.load_initial_variables(args, args.dataset_lst_filename)
    device_latency_lst = Utils.generate_device_latency(args)
    single_grad_latency_lst = Utils.generate_grad_train_latency(args)

    init_Rl = RL_selector((1, 5), 2)

    R_lst = [dict() for i in range(args.N)]
    #print([len(dataset_lst[i]) for i in range(args.N)])
    status_lst = [-1 for i in range(args.N)]
    Q_lst = [dict() for i in range(args.N)]

    gradients_lst = []
    time_lst = []
    acc_lst = []
    staleness_lst = []

    # 存放本地模型权重
    local_weights_store = simpy.Store(env, capacity=args.N)

    if train:
        # 记录初始准确率
        #sample_indexes = dataset_lst[0]["test"]
        #test_data = dataset.sample_test_datas(sample_indexes)
        test_data = dataset.sample_test_datas()
        global_model.set_weights(w_global)
        loss, acc = test_cnn(global_model, test_data)

        gradients_lst.append(gradients_count)
        time_lst.append(0)
        acc_lst.append(acc)
        staleness_lst.append(0)
        print(f"initial accuracy: {acc}")

    server_info = {'round': 0, 'gradients_count': gradients_count,
                   'end': env.now, 'staleness': 0}
    server_info_lst.append(server_info)

    env.process(scheduler(env, train, dataset, dataset_lst,
                device_latency_lst, local_weights_store, worker_info_lst, args))
    env.process(updater(env, train, start_time, local_weights_store, dataset, dataset_lst,
                gradients_lst, time_lst, acc_lst, staleness_lst, server_info_lst, custom, args))
    env.run()


def start_train(args, custom, gpu):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    train = True
    worker_info_lst, server_info_lst = [], []
    run(train, worker_info_lst, server_info_lst, custom, args)
    return worker_info_lst, server_info_lst


def start_time_simulation(args, custom):
    train = False
    worker_info_lst, server_info_lst = [], []
    run(train, worker_info_lst, server_info_lst, custom, args)
    return worker_info_lst, server_info_lst


if __name__ == "__main__":
    args = args_parser()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    custom = {
        "method": "FedAmdSampleRL",
        # "T": 1000000,
        "fraction": 0.2,
        #"regularization": True,
        "regularization": True,
        "trigger_number": 1,
        "trigger_interval": 5,
        "S": 99,
        #"alpha": 0.4,
        "c": 0.33,
        "d": 0.66,
        "beta": 0.99,
        # "gamma": 5,
    }
    # 根据custom来更新args
    args = Utils.update_args(args, custom)
    random.seed(args.seed)

    print(args)
    if args.wandb > 0:
        wandb.init(project="federated learning " + args.model + " " + args.dataset + " " + str(args.local_lr), name="fedamd", config=args)

    train = True
    gpu = args.gpu
    if train:
        start = time.time()
        worker_info_lst, server_info_lst = start_train(args, custom, gpu)
        print(f'total time used: {time.time() - start}')
        Utils.show_gantt(worker_info_lst, server_info_lst, custom, args)
    else:
        worker_info_lst, server_info_lst = start_time_simulation(args, custom)
        Utils.show_gantt(worker_info_lst, server_info_lst, custom, args)
