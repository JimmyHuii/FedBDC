import copy
import math

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from flearn.models import cnn
from scipy.spatial.distance import jensenshannon
from collections import defaultdict

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0][1])
    total = 0
    for i in range(0, len(w)):
        total += w[i][0]
    for key in w_avg.keys():
        w_avg[key] *= w[0][0]
        for i in range(1, len(w)):
            w_avg[key] += w[i][1][key] * w[i][0]
        w_avg[key] = torch.div(w_avg[key], total)
    return w_avg

def average_weights_new(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0][1])
    total = len(w)
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][1][key]
        w_avg[key] = torch.div(w_avg[key], total)
    return w_avg

def scale_weights(w, factor):
    """
    Scales the values of a weight dictionary by a given factor.
    """
    scaled_w = {}
    for key in w.keys():
        scaled_w[key] = w[key] * factor
    return scaled_w


def ratio_combine(w1, w2, ratio=0):
    """
    将两个权重进行加权平均，ratio 表示 w2 的占比
    :param w1:
    :param w2:
    :param ratio:
    :return:
    """
    w = copy.deepcopy(w1)
    for key in w.keys():
        w[key] = (w2[key] - w1[key]) * ratio + w1[key]
    return w

def ratio_minus(w1, P, ratio=0):
    w = copy.deepcopy(w1)
    for key in w.keys():
        w[key] = w1[key] - P[key] * ratio
    return w

def test_inference(model, test_dataset, device):
    """
    Returns the test accuracy and loss.
    """
    model.eval() #model.eval()的作用是不启用 Batch Normalization 和 Dropout，model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1) #torch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一列的列索引）
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item() #若predicted与label.data对应数值相同，则torch.eq()返回1，否则返回0，张量类型
                                                                        #torch.eq(predicted,label.data).sum() 返回一个张量，张量值为对应值相同的个数
        total += len(labels)

    accuracy = correct/total
    return round(accuracy, 4), round(loss / (len(testloader)), 4)

def diff_model(w1, w2): #w->dict
    # 计算两个模型各层的差异
    diff = []
    preNum = 0
    for key in w1.keys():
        a = (torch.abs(w2[key] - w1[key]))
        b = torch.mean(a)
        # c = torch.sum(a)
        if "weight" in key:
            diff.append(round(b.item(), 6))
            preNum = a.numel()
        elif "bias" in key:
            diff[-1] = round((diff[-1] * preNum + b.item() * a.numel()) / (preNum + a.numel()), 6)#why?
    return diff

# def get_model_like_tensor(model, dtype=torch.float32):
#     # 获得和模型一样维度的tensor空向量
#     # 我们和模型一样用字典来保存这些tensor空向量，首先遍历模型的名字，然后每个名字对应的tensor
#     cloned_tensor = {k: torch.zeros_like(v, dtype=dtype) #生成和括号内变量维度维度一致的全是零的内容
#                     for k, v in model.named_parameters()
#                     if v.requires_grad}
#     return cloned_tensor


def get_model_like_tensor(model, dtype=torch.float32):
    """
    获得和模型一样维度的tensor空向量
    (修正版：迭代 state_dict() 来包含参数和缓冲区)
    (修正版2：重新接受可选的 dtype 参数)
    """

    # 1. 迭代 model.state_dict().items() 来获取 *所有* 键
    #    (包括 weight, bias, running_mean, running_var, num_batches_tracked)

    # 2. torch.zeros_like 会智能地处理 dtype:
    #    - 如果 dtype=None (默认), 它会保留 v 本身的类型 (例如, weight 是 float32, num_batches_tracked 是 int64)
    #    - 如果 dtype=torch.int64 (您的掩码调用), 它会将 *所有* 张量都创建为 int64, 这正是您想要的
    cloned_tensor = {k: torch.zeros_like(v, dtype=dtype)
                     for k, v in model.state_dict().items()}

    return cloned_tensor

def update_model_stability(last_w, cur_w, E, E_abs, mask, alpha=0.99):
    # 更新模型稳定性状态
    # 首先更新 E 和 E_abs,具体方法就是逐层计算了
    w_delta = copy.deepcopy(cur_w)
    for key in w_delta.keys():
        w_delta[key] = w_delta[key] - last_w[key]
        E[key] = alpha * E[key] + (1 - alpha) * w_delta[key]
        E_abs[key] = alpha * E_abs[key] + (1 - alpha) * torch.abs(w_delta[key])

def ratio_mask(E, E_abs, mask, ratio=0.8):
    p = copy.deepcopy(E)
    for key in p.keys():  # 计算稳定性
        p[key] = torch.abs(p[key]) / E_abs[key]
        p[key] = torch.where(torch.isnan(p[key]), torch.zeros_like(p[key]), p[key])  # ？

    p_flat = torch.cat([v.flatten() for k, v in p.items()])  # 把字典p中的数据拼接
    number_of_weights_to_prune = int(np.ceil(ratio * p_flat.shape[0]))  # 确定要上传的数据的个数
    threshold = torch.sort(torch.abs(p_flat))[0][number_of_weights_to_prune]  # 将p中的数据从小到大排序后以ratio将某个数确定为阈值

    for key in p.keys():  # 可将一整个tensor与某一值比较
        mask[key] = (p[key] <= threshold) * 1

def threshold_mask(E, E_abs, mask, threshold=0.05):
    p = copy.deepcopy(E)
    for key in p.keys():
        p[key] = torch.abs(p[key]) / E_abs[key]
        p[key] = torch.where(torch.isnan(p[key]), torch.zeros_like(p[key]), p[key])
        mask[key] = (p[key] <= threshold) * 1

def update_model_stability_ignore_frozen(last_w, cur_w, E, E_abs, mask,
                                         L, I, cur_epoch, freq, alpha=0.99, threshold=0.05):
    # 更新模型稳定性状态，不考虑冻结的参数，同时考虑一个冻结时间
    # 首先更新 E 和 E_abs,具体方法就是逐层计算了
    # 之后计算 P，获得来求得 mask
    # L 要冻结的时间长度， I 要冻结的结束轮数
    w_delta = copy.deepcopy(cur_w)
    for key in w_delta.keys():
        w_delta[key] = w_delta[key] - last_w[key]
        E[key] = torch.where(mask[key] == 1, E[key],  alpha * E[key] + (1-alpha) * w_delta[key])
        E_abs[key] = torch.where(mask[key] == 1, E_abs[key],  alpha * E_abs[key] + (1-alpha) * torch.abs(w_delta[key]))

    p = copy.deepcopy(E)
    for key in p.keys():
        p[key] = torch.abs(p[key]) / E_abs[key]
        p[key] = torch.where(torch.isnan(p[key]), torch.zeros_like(p[key]), p[key])
        L[key] = torch.where(p[key] <= threshold,  L[key] + freq, L[key] / 2)
        I[key] = torch.where(mask[key] == 1, I[key], cur_epoch + L[key])
        mask[key] = (cur_epoch + 0.5 < I[key]) * 1


def update_model_stability_ignore_frozen_by_ratio(last_w, cur_w, E, E_abs, mask,
                                         L, I, cur_epoch, freq, alpha=0.99, ratio=0.2):
    # 更新模型稳定性状态，不考虑冻结的参数，同时考虑一个冻结时间
    # 首先更新 E 和 E_abs,具体方法就是逐层计算了
    # 之后计算 P，获得来求得 mask
    # L 要冻结的时间长度， I 要冻结的结束轮数
    w_delta = copy.deepcopy(cur_w)
    for key in w_delta.keys():
        w_delta[key] = w_delta[key] - last_w[key]
        E[key] = torch.where(mask[key] == 1, E[key],  alpha * E[key] + (1-alpha) * w_delta[key])
        E_abs[key] = torch.where(mask[key] == 1, E_abs[key],  alpha * E_abs[key] + (1-alpha) * torch.abs(w_delta[key]))

    p = copy.deepcopy(E)
    for key in p.keys():
        p[key] = torch.abs(p[key]) / E_abs[key]
        p[key] = torch.where(torch.isnan(p[key]), torch.zeros_like(p[key]), p[key])

    p_flat = torch.cat([v.flatten() for k, v in p.items()])
    number_of_weights_to_prune = int(np.ceil(ratio * p_flat.shape[0]))
    threshold = torch.sort(torch.abs(p_flat))[0][number_of_weights_to_prune]

    for key in p.keys():
        L[key] = torch.where(p[key] <= threshold,  L[key] + freq, L[key] / 2)
        I[key] = torch.where(mask[key] == 1, I[key], cur_epoch + L[key])
        mask[key] = (cur_epoch + 0.5 < I[key]) * 1

# def topK(record_w, cur_w, ratio, record_mask):
#     w_delta = copy.deepcopy(cur_w)
#     for key in w_delta.keys():
#         w_delta[key] = torch.abs(w_delta[key] - record_w[key])
#
#     w_delta_flat = torch.cat([v.flatten() for k, v in w_delta.items()])
#     number_of_weights_to_filter = int(np.ceil(ratio * w_delta_flat.shape[0]))
#     threshold = torch.sort(torch.abs(w_delta_flat))[0][number_of_weights_to_filter]
#
#     for key in w_delta.keys():
#         record_mask[key] = torch.where(w_delta[key] <= threshold, 1, 0)

# def topK(record_w, cur_w, ratio, record_mask):
#     common_keys = set(record_w.keys()).intersection(set(cur_w.keys()))
#
#     w_delta = copy.deepcopy(cur_w)
#     for key in common_keys:
#         w_delta[key] = torch.abs(w_delta[key] - record_w[key])
#
#     w_delta_flat = torch.cat([v.flatten() for k, v in w_delta.items()])
#     number_of_weights_to_filter = int(np.ceil(ratio * w_delta_flat.shape[0]))
#     threshold = torch.sort(torch.abs(w_delta_flat))[0][number_of_weights_to_filter]
#
#     for key in common_keys:
#         record_mask[key] = torch.where(w_delta[key] <= threshold, 1, 0)


def topK(record_w: dict, cur_w: dict, ratio: float, record_mask: dict):
    """
    一个混合 Top-K 函数，它能正确处理参数 (Parameters) 和缓冲区 (Buffers)。

    - Top-K 压缩 *只* 应用于 'weight' 和 'bias' 键。
    - 'running_mean', 'running_var' 等缓冲区键 *总是* 被发送。

    Args:
        record_w (dict): 训练前的 state_dict (完整的, 包含缓冲区)
        cur_w (dict): 训练后的 state_dict (完整的, 包含缓冲区)
        ratio (float): 要 *过滤掉* (不发送) 的比例 (例如 0.4, 表示过滤掉 40%)
        record_mask (dict): 要被就地修改的掩码字典。
                            (0 = 发送, 1 = 不发送)
    """

    # --- 1. 将掩码中的键分为 "参数" 和 "缓冲区" ---
    param_keys = []
    buffer_keys = []
    # for key in record_mask.keys():
    #     if 'weight' in key or 'bias' in key:
    #         param_keys.append(key)
    #     else:
    #         # 这将捕获 'running_mean', 'running_var', 'num_batches_tracked'
    #         buffer_keys.append(key)
    for key in record_mask.keys():

        # --- 这是新的、更精确的过滤规则 ---

        # 规则 1: 检查是否是 BN 层
        # (您的实现中，主干BN层叫 'bn1', 'bn2'等, 快捷BN层叫 'shortcut.1')
        is_bn_layer = ('bn' in key) or ('shortcut.1' in key)

        # 规则 2: 检查是否是 缓冲区 (Buffers)
        is_buffer_data = ('running' in key) or ('tracked' in key)

        # --- 决策 ---

        if is_bn_layer or is_buffer_data:
            # 如果是 BN 层的 *任何* 部分 (weight, bias, running_mean)
            # 或者 *任何* 缓冲区数据
            # -> 归入 "全量发送" 列表
            buffer_keys.append(key)

        elif 'weight' in key or 'bias' in key:
            # 如果不是 BN 层, 并且是 weight/bias
            # (这必然是 'conv', 'shortcut.0', 或 'linear'/'fc' 层)
            # -> 归入 "Top-K" 列表
            param_keys.append(key)

        else:
            # 备用：其他未知的键 (安全起见，全量发送)
            buffer_keys.append(key)

    # --- 2. *只* 计算 "参数" 的变化量 ---
    w_delta_params = {}
    for key in param_keys:
        if key in record_w and key in cur_w:
            w_delta_params[key] = torch.abs(cur_w[key] - record_w[key])
        else:
            # 如果参数不存在于模型中 (不应该发生), 默认变化量为0
            w_delta_params[key] = torch.zeros_like(record_mask[key])

    # --- 3. *只* 使用 "参数" 变化量来计算 Top-K 阈值 ---
    if not w_delta_params:
        print("警告 (topK_hybrid): 找不到任何 'weight' 或 'bias' 键来计算 Top-K。")
        # 将所有键设置为 0 (发送) 作为安全的回退
        for key in record_mask.keys():
            record_mask[key] = 0
        return

    # 将所有 *参数* 变化量展平
    w_delta_flat = torch.cat([v.flatten() for k, v in w_delta_params.items()])

    # 计算 40% (ratio=0.4) 对应的索引
    number_of_weights_to_filter = int(np.ceil(ratio * w_delta_flat.shape[0]))

    threshold = 0.0
    if number_of_weights_to_filter > 0 and len(w_delta_flat) > 0:
        # 确保索引不越界
        if number_of_weights_to_filter >= len(w_delta_flat):
            threshold = torch.max(w_delta_flat)  # 过滤所有
        else:
            # 找到第 40% 小的值作为阈值
            threshold = torch.sort(w_delta_flat)[0][number_of_weights_to_filter]

    # --- 4. 智能地应用掩码 ---

    # (A) 将 Top-K 规则应用于 "参数"
    for key in param_keys:
        # 变化量 <= 阈值 (Bottom 40%) -> 1 (不发送)
        # 变化量 > 阈值 (Top 60%)   -> 0 (发送)
        record_mask[key] = torch.where(w_delta_params[key] <= threshold, 1, 0)

    # (B) 将 "始终发送" 规则应用于 "缓冲区"
    for key in buffer_keys:
        # 0 = 发送
        record_mask[key] = torch.zeros_like(record_mask[key], dtype=torch.int)  # 始终发送 'running_mean', 'running_var' 等


# def topK_layerwise(record_w, cur_w, ratio, record_mask):
#     """
#     从最后一层开始逐层向前选择参数的 Top-K 掩码生成函数。
#     - ratio: 传输比例 (例如 0.4 表示传输 40% 的参数)。
#     - record_mask: 掩码字典，被选择的参数位置置 0 (传输)，未选置 1 (不传输)。
#     """
#     common_keys = list(set(record_w.keys()).intersection(set(cur_w.keys())))  # 转换为列表以保持顺序
#
#     # 计算参数差
#     w_delta = copy.deepcopy(cur_w)
#     for key in common_keys:
#         w_delta[key] = torch.abs(w_delta[key] - record_w[key])
#
#     # 计算总参数量和需要传输的参数数量
#     total_params = sum(v.numel() for v in w_delta.values())
#     number_to_select = int(ratio * total_params)  # 需要传输的参数数量
#
#     # 已选参数量
#     selected_count = 0
#
#     # 从最后一层开始逐层向前
#     for key in reversed(common_keys):
#         layer_delta = w_delta[key]
#         layer_params = layer_delta.numel()
#
#         remaining_to_select = number_to_select - selected_count
#
#         if remaining_to_select <= 0:
#             # 已达到传输数量，停止
#             break
#
#         if layer_params <= remaining_to_select:
#             # 全选该层（传输全部参数）
#             record_mask[key] = torch.zeros_like(layer_delta, dtype=torch.int)
#             selected_count += layer_params
#         else:
#             # 该层参数量 > 剩余需要传输的数量，使用 Top-K 选择剩余数量的参数
#             layer_delta_flat = layer_delta.flatten()
#             k_i = remaining_to_select
#             # 使用 topk 选择最大的 k_i 个位置
#             values, indices = torch.topk(torch.abs(layer_delta_flat), k_i, sorted=False)
#
#             # 生成掩码：全置 1 (不传输)，然后在选中的 indices 置 0 (传输)
#             record_mask[key] = torch.ones_like(layer_delta, dtype=torch.int)
#             record_mask[key].flatten()[indices] = 0
#
#             selected_count += k_i
#             # 达到传输数量，停止
#             break
#
#     # 对于未处理的层（前面的层），全置 1 (不传输)
#     for key in common_keys:
#         if key not in record_mask:
#             record_mask[key] = torch.ones_like(w_delta[key], dtype=torch.int)
#
#     return record_mask

def topK_layerwise(record_w: dict, cur_w: dict, ratio: float, record_mask: dict):
    """
    一个混合的、逐层 Top-K 函数，它能正确处理参数 (Parameters) 和缓冲区 (Buffers)。

    - "ratio" (例如 0.4) 的预算 *只* 应用于 'weight' 和 'bias' 键。
    - Top-K 选择逻辑从后向前 *只* 应用于 'weight' 和 'bias' 键。
    - 'running_mean', 'running_var' 等缓冲区键 *总是* 被发送。

    Args:
        record_w (dict): 训练前的 state_dict (完整的, 包含缓冲区)
        cur_w (dict): 训练后的 state_dict (完整的, 包含缓冲区)
        ratio (float): 要 *传输* 的比例 (例如 0.4, 表示传输 40% 的 *参数*)
        record_mask (dict): 要被就地修改的掩码字典。
                            (0 = 发送, 1 = 不发送)
    """

    # --- 1. 将掩码中的键分为 "参数" 和 "缓冲区" ---
    param_keys = []
    buffer_keys = []
    # 我们必须使用 record_mask.keys() 作为“模板”，因为它保证是完整的
    all_keys = list(record_mask.keys())

    for key in record_mask.keys():

        # --- 这是新的、更精确的过滤规则 ---

        # 规则 1: 检查是否是 BN 层
        # (您的实现中，主干BN层叫 'bn1', 'bn2'等, 快捷BN层叫 'shortcut.1')
        is_bn_layer = ('bn' in key) or ('shortcut.1' in key)

        # 规则 2: 检查是否是 缓冲区 (Buffers)
        is_buffer_data = ('running' in key) or ('tracked' in key)

        # --- 决策 ---

        if is_bn_layer or is_buffer_data:
            # 如果是 BN 层的 *任何* 部分 (weight, bias, running_mean)
            # 或者 *任何* 缓冲区数据
            # -> 归入 "全量发送" 列表
            buffer_keys.append(key)

        elif 'weight' in key or 'bias' in key:
            # 如果不是 BN 层, 并且是 weight/bias
            # (这必然是 'conv', 'shortcut.0', 或 'linear'/'fc' 层)
            # -> 归入 "Top-K" 列表
            param_keys.append(key)

        else:
            # 备用：其他未知的键 (安全起见，全量发送)
            buffer_keys.append(key)

    # --- 2. *只* 计算 "参数" 的变化量 ---
    w_delta_params = {}
    for key in param_keys:
        if key in record_w and key in cur_w:
            w_delta_params[key] = torch.abs(cur_w[key] - record_w[key])
        else:
            w_delta_params[key] = torch.zeros_like(record_mask[key])  # 默认变化量为0

    # --- 3. *只* 基于 "参数" 计算总预算 ---
    total_params = sum(v.numel() for v in w_delta_params.values())
    if total_params == 0:
        # 没有任何参数, 这种情况不应该发生, 但作为安全检查
        print("警告 (topK_layerwise_hybrid): 在模型中找不到任何 'weight' 或 'bias'。")
        number_to_select = 0
    else:
        number_to_select = int(ratio * total_params)  # 需要传输的 *参数* 总量

    selected_count = 0

    # --- 4. 智能地应用掩码 ---

    # (A) 首先, 默认将 *所有参数* 标记为 1 (不发送)
    for key in param_keys:
        record_mask[key] = torch.ones_like(record_mask[key], dtype=torch.int)

    # (B) 从后向前遍历 *参数* 键，"花费" 我们的预算
    for key in reversed(param_keys):
        layer_delta = w_delta_params[key]
        layer_params = layer_delta.numel()

        if layer_params == 0:
            continue  # 跳过空层

        remaining_to_select = number_to_select - selected_count

        if remaining_to_select <= 0:
            # 预算已用完。
            # 所有更早的层 (包括当前层) 将保持为 1 (不发送)。
            break

        if layer_params <= remaining_to_select:
            # 预算充足, 全选该层 (传输全部参数)
            record_mask[key] = torch.zeros_like(layer_delta, dtype=torch.int)
            selected_count += layer_params
        else:
            # 预算不足, 该层是最后一层。
            # 在该层内使用 Top-K 选择剩余数量的参数
            layer_delta_flat = layer_delta.flatten()
            k_i = remaining_to_select  # 剩余的预算

            # 使用 topk 选择最大的 k_i 个位置
            values, indices = torch.topk(torch.abs(layer_delta_flat), k_i, sorted=False)

            # 掩码已经是 1 (不传输), 只在选中的 indices 处 V置 0 (传输)
            record_mask[key].flatten()[indices] = 0

            selected_count += k_i
            # 预算已用完，停止
            break

    # (C) 最后, 将 *所有缓冲区* 标记为 0 (始终发送)
    for key in buffer_keys:
        record_mask[key] = torch.zeros_like(record_mask[key], dtype=torch.int)

    return record_mask  # record_mask 已被就地修改


def _group_params_by_block(param_keys: list) -> dict:
    """
    辅助函数：将参数键按“块”分组。
    这与您的绘图脚本中的聚合逻辑一致。
    例如:
     - 'features.0.weight' -> 'features.0'
     - 'layer1.0.conv1.weight' -> 'layer1.0'
     - 'linear.weight' -> 'linear'
    """
    groups = defaultdict(list)
    for key in param_keys:
        parts = key.split('.')

        # 1. 识别 ResNet 的残差块 (例如 layer1.0)
        if key.startswith('layer') and len(parts) >= 2:
            # 取前两部分，组合成 'layer1.0' 这样的块名称
            block_name = ".".join(parts[:2])

        # 2. 其他所有情况 (包括 conv1, linear, fc, features.0, classifier.2 等)
        else:
            if '.' in key:
                # 例如: 'features.0.weight' -> 'features.0'
                block_name = ".".join(parts[:-1])
            else:
                block_name = key

        groups[block_name].append(key)
    return groups


def topK_layerwise_new(record_w: dict, cur_w: dict, ratio: float, record_mask: dict,
                       priority_order: list = None):
    """
    (新版) 混合逐层Top-K函数，支持自定义优先级顺序。

    1. 自动将所有 BN/缓冲区 键设置为 0 (全量发送)。
    2. 计算 Top-K *只* 基于 'weight'/'bias' (参数)。
    3. 如果提供了 'priority_order' (块/层名称列表):
       - 优先按该列表顺序分配预算。
       - 然后按“从后向前”的顺序分配剩余预算给其他参数。
    4. 如果 'priority_order' 为 None:
       - 恢复为默认的“从后向前”的顺序。

    Args:
        record_w (dict): 训练前的 state_dict (完整)
        cur_w (dict): 训练后的 state_dict (完整)
        ratio (float): 要 *传输* 的比例 (例如 0.4, 传输 40% 的 *参数*)
        record_mask (dict): 要被就地修改的掩码字典 (0 = 发送, 1 = 不发送)
        priority_order (list, optional):
            一个字符串列表，包含*优先*处理的“块”名称。
            例如: ['classifier.2', 'features.0']
    """

    # --- 1. 将掩码中的键分为 "参数" 和 "缓冲区" ---
    param_keys = []
    buffer_keys = []
    all_keys = list(record_mask.keys())

    for key in all_keys:
        # (我们使用您最终确认的过滤逻辑)
        is_bn_layer = ('bn' in key) or ('shortcut.1' in key) or ('downsample.1' in key)
        is_buffer_data = ('running' in key) or ('tracked' in key)

        if is_bn_layer or is_buffer_data:
            buffer_keys.append(key)
        elif 'weight' in key or 'bias' in key:
            param_keys.append(key)
        else:
            buffer_keys.append(key)  # 其他未知的键 (安全起见，全量发送)

    # --- 2. *只* 计算 "参数" 的变化量 ---
    w_delta_params = {key: torch.zeros_like(record_mask[key]) for key in param_keys}
    for key in param_keys:
        if key in record_w and key in cur_w:
            w_delta_params[key] = torch.abs(cur_w[key] - record_w[key])

    # --- 3. *只* 基于 "参数" 计算总预算 ---
    total_params = sum(v.numel() for v in w_delta_params.values())
    number_to_select = 0
    if total_params > 0:
        number_to_select = int(ratio * total_params)

    # --- 4. [新逻辑] 构建自定义处理顺序 ---

    # 4.1 按“块”分组所有参数
    # block_map -> {'features.0': ['features.0.weight', 'features.0.bias'], ...}
    block_map = _group_params_by_block(param_keys)

    # 4.2 计算处理顺序
    processing_order_keys = []

    if priority_order:
        print("--- (Top-K) 应用自定义层优先级 ---")
        # 1. 按给定顺序添加 "优先" 的键
        priority_blocks_found = []
        for block_name in priority_order:
            if block_name in block_map:
                processing_order_keys.extend(block_map[block_name])
                priority_blocks_found.append(block_name)

        # 2. 按 "从后向前" 的顺序添加 "剩余" 的键
        #    (reversed(block_map.keys()) 提供了默认的从后向前顺序)
        fallback_keys = []
        for block_name in reversed(block_map.keys()):
            if block_name not in priority_blocks_found:
                fallback_keys.extend(block_map[block_name])

        processing_order_keys.extend(fallback_keys)

    else:
        # 默认行为：按 "从后向前" 的顺序处理所有参数
        print("\n--- (Top-K) 应用默认 (从后向前) 顺序 ---")
        processing_order_keys = list(reversed(param_keys))

    # --- 5. 智能地应用掩码 ---

    # (A) 默认将 *所有参数* 标记为 1 (不发送)
    for key in param_keys:
        record_mask[key] = torch.ones_like(record_mask[key], dtype=torch.int)

    # (B) 按照我们新构建的 "processing_order_keys" 顺序，"花费" 我们的预算
    selected_count = 0
    for key in processing_order_keys:
        layer_delta = w_delta_params[key]
        layer_params = layer_delta.numel()

        if layer_params == 0: continue
        remaining_to_select = number_to_select - selected_count
        if remaining_to_select <= 0:
            break  # 预算已用完

        if layer_params <= remaining_to_select:
            # 预算充足, 全选该层
            record_mask[key] = torch.zeros_like(layer_delta, dtype=torch.int)
            selected_count += layer_params
        else:
            # 预算不足, 该层是最后一层
            layer_delta_flat = layer_delta.flatten()
            k_i = remaining_to_select
            values, indices = torch.topk(torch.abs(layer_delta_flat), k_i, sorted=False)
            record_mask[key].flatten()[indices] = 0
            selected_count += k_i
            break

    # (C) 最后, 将 *所有缓冲区* 标记为 0 (始终发送)
    for key in buffer_keys:
        record_mask[key] = torch.zeros_like(record_mask[key], dtype=torch.int)

    return record_mask


def topK_custom_hybrid(record_w: dict, cur_w: dict, ratio: float, record_mask: dict,
                       priority_order: list = None):
    """
    (最新版) 自定义混合策略：锚点层全选 + 剩余层全局Top-K。

    逻辑：
    1. [缓冲区] BN 层/缓冲区 -> 始终全量发送 (mask=0)。
    2. [锚点层] `priority_order` 中的层 -> 优先全量发送 (直至预算耗尽)。
    3. [剩余层] 所有其他层 -> 使用剩余预算进行 **Global Top-K** 选择。

    Args:
        ratio: 传输比例 (例如 0.5 表示传输 50% 的参数)。
        priority_order: 优先处理的层名称列表 (例如 ['linear', 'conv1'])。
    """

    # --- 1. 分类：分离 "参数" 和 "缓冲区" ---
    param_keys = []
    buffer_keys = []
    all_keys = list(record_mask.keys())

    for key in all_keys:
        # 过滤逻辑 (BN/Buffer)
        is_bn_layer = ('bn' in key) or ('shortcut.1' in key) or ('downsample.1' in key)
        is_buffer_data = ('running' in key) or ('tracked' in key)

        if is_bn_layer or is_buffer_data:
            buffer_keys.append(key)
        elif 'weight' in key or 'bias' in key:
            param_keys.append(key)
        else:
            buffer_keys.append(key)

            # --- 2. *只* 计算 "参数" 的变化量 ---
    w_delta_params = {key: torch.zeros_like(record_mask[key]) for key in param_keys}
    for key in param_keys:
        if key in record_w and key in cur_w:
            w_delta_params[key] = torch.abs(cur_w[key] - record_w[key])

    # --- 3. 计算总预算 ---
    total_params_count = sum(v.numel() for v in w_delta_params.values())
    total_budget = int(ratio * total_params_count)
    remaining_budget = total_budget

    # --- 4. 识别 "锚点层" 和 "剩余层" ---
    block_map = _group_params_by_block(param_keys)

    # 找出哪些 key 属于优先列表 (Anchor Keys)
    anchor_keys = []
    processed_blocks = set()

    if priority_order:
        for block_name in priority_order:
            if block_name in block_map:
                anchor_keys.extend(block_map[block_name])
                processed_blocks.add(block_name)

    # 找出剩下的 key (Rest Keys)
    rest_keys = []
    for block_name, keys in block_map.items():
        if block_name not in processed_blocks:
            rest_keys.extend(keys)

    # --- 5. 执行选择 ---

    # (A) 初始化：默认不发送 (置 1)
    for key in param_keys:
        record_mask[key] = torch.ones_like(record_mask[key], dtype=torch.int)

    # (B) 第一阶段：处理锚点层 (按顺序贪婪填充)
    # 我们按照 priority_order 的顺序来，以防万一预算连锚点层都填不满

    # 为了按顺序处理，我们需要重新遍历 priority_order
    if priority_order:
        for block_name in priority_order:
            if block_name not in block_map: continue

            # 获取该块的所有参数键
            block_keys = block_map[block_name]

            for key in block_keys:
                if remaining_budget <= 0: break  # 预算耗尽

                num_params = w_delta_params[key].numel()

                if num_params <= remaining_budget:
                    # 预算够：全选该参数
                    record_mask[key] = torch.zeros_like(record_mask[key], dtype=torch.int)
                    remaining_budget -= num_params
                else:
                    # 预算不够：填满预算后停止 (在该参数内部 Top-K)
                    flat_delta = w_delta_params[key].flatten()
                    _, indices = torch.topk(flat_delta, remaining_budget, sorted=False)
                    record_mask[key].flatten()[indices] = 0
                    remaining_budget = 0
                    break

            if remaining_budget <= 0: break

    # (C) 第二阶段：处理剩余层 (Global Top-K)
    # 只有当还有预算，且还有剩余层时才执行
    if remaining_budget > 0 and rest_keys:
        # 1. 收集所有剩余层的变化量到一个大池子里
        rest_deltas = []
        for key in rest_keys:
            rest_deltas.append(w_delta_params[key].flatten())

        if rest_deltas:
            # 拼接成一个巨大的向量
            all_rest_deltas = torch.cat(rest_deltas)

            # 2. 如果剩余参数总量 < 剩余预算，全选
            if all_rest_deltas.numel() <= remaining_budget:
                for key in rest_keys:
                    record_mask[key] = torch.zeros_like(record_mask[key], dtype=torch.int)
            else:
                # 3. 否则，执行 Global Top-K
                # 找到阈值
                threshold_val = torch.kthvalue(all_rest_deltas, all_rest_deltas.numel() - remaining_budget + 1).values
                # 或者用 topk (对于极大向量，kthvalue 可能更省内存，或者 sort)
                # 这里为了稳健使用 sort
                # threshold_val = torch.sort(all_rest_deltas)[0][all_rest_deltas.numel() - remaining_budget]

                # 应用阈值
                for key in rest_keys:
                    # 大于阈值的置 0 (发送)
                    record_mask[key] = torch.where(w_delta_params[key] >= threshold_val, 0, 1)

    # (D) 缓冲区：全量发送 (置 0)
    for key in buffer_keys:
        record_mask[key] = torch.zeros_like(record_mask[key], dtype=torch.int)

    return record_mask


def topK_hypernet(record_w: dict, cur_w: dict, alpha_map: dict, record_mask: dict):
    """
    基于 HyperNetwork 输出的 Block-wise Alpha 进行参数选择。

    Args:
        record_w (dict): 上一轮的模型权重 (Previous Global Model)
        cur_w (dict): 当前训练后的模型权重 (Current Local Model)
        alpha_map (dict): 块名到压缩率的映射 { 'layer1.0': 0.5, 'conv1': 0.8 ... }
                          (由 HyperNet 生成并传给客户端)
        record_mask (dict): 要被就地修改的掩码字典 (0 = 发送, 1 = 不发送)
    """

    # --- 1. 遍历所有键进行分类处理 ---
    all_keys = list(record_mask.keys())

    for key in all_keys:
        # ---------------------------------------------------------
        # 规则 A: 缓冲区与 BN 层 -> 始终全量发送 (Mask = 0)
        # ---------------------------------------------------------
        # (与你之前的逻辑保持严格一致)
        is_bn_layer = ('bn' in key) or ('shortcut.1' in key) or ('downsample.1' in key)
        is_buffer_data = ('running' in key) or ('tracked' in key)

        if is_bn_layer or is_buffer_data:
            # 0 代表发送 (Keep)
            record_mask[key] = torch.zeros_like(record_mask[key], dtype=torch.int)
            continue  # 处理下一个键

        # ---------------------------------------------------------
        # 规则 B: 权重参数 (Weight/Bias) -> 根据 Alpha 进行 Top-K
        # ---------------------------------------------------------
        if 'weight' in key or 'bias' in key:
            # 1. 计算变化量 (Importance)
            if key in record_w and key in cur_w:
                delta = torch.abs(cur_w[key] - record_w[key])
            else:
                # 防御性代码：如果找不到对应键，默认全传
                record_mask[key] = torch.zeros_like(record_mask[key], dtype=torch.int)
                continue

            # 2. 解析 Block Name (找到这个参数对应的 alpha)
            # 逻辑需与 get_model_blocks 保持一致
            parts = key.split('.')
            if key.startswith('layer') and len(parts) >= 2:
                block_name = ".".join(parts[:2])  # 例如 layer1.0
            elif '.' in key:
                block_name = ".".join(parts[:-1])  # 例如 features.0
            else:
                block_name = key  # 例如 linear

            # 3. 获取压缩率并计算 K
            if block_name in alpha_map:
                ratio = alpha_map[block_name]

                # 如果是 Tensor (GPU/CPU)，取 item() 转为 float
                if isinstance(ratio, torch.Tensor):
                    ratio = ratio.item()

                num_elements = delta.numel()
                k = int(ratio * num_elements)

                # 边界保护：至少传 1 个，最多全传
                k = max(1, min(k, num_elements))

                # 4. 执行 Top-K 选择
                if k == num_elements:
                    # 全选
                    record_mask[key] = torch.zeros_like(record_mask[key], dtype=torch.int)
                else:
                    # 找到第 k 大的阈值
                    # view(-1) 展平 -> kthvalue 找第 (N-k+1) 小的值 (即第 k 大)
                    flatten_delta = delta.view(-1)
                    threshold = flatten_delta.kthvalue(num_elements - k + 1).values

                    # 大于等于阈值的置 0 (发送)，小于的置 1 (不发送)
                    record_mask[key] = torch.where(delta >= threshold, 0, 1)
            else:
                # 如果 HyperNet 没覆盖这个层 (比如 Bias 没在 map 里)，默认全传
                record_mask[key] = torch.zeros_like(record_mask[key], dtype=torch.int)

        else:
            # 其他未知类型的键，默认全传
            record_mask[key] = torch.zeros_like(record_mask[key], dtype=torch.int)

    return record_mask

def calculate_js_divergence_for_models(model_before: nn.Module,
                                       model_after: nn.Module,
                                       num_bins: int = 100) -> dict:
    """
    计算两个PyTorch模型之间对应层的所有参数（包括weight和bias）的JS散度。

    Args:
        model_before (nn.Module): 训练前的模型。
        model_after (nn.Module): 训练后的模型。
        num_bins (int): 构建直方图时使用的“箱子”数量。

    Returns:
        dict: 一个字典，键是参数的名称，值是该参数在训练前后的JS散度。
    """
    js_divergences = {}

    # 获取两个模型的所有参数，并确保它们一一对应
    params_before = dict(model_before.named_parameters())
    params_after = dict(model_after.named_parameters())

    print("开始计算JS散度，将分析以下所有参数：")

    # --- 改进点 ---
    # 不再筛选 'weight'，而是遍历所有找到的参数
    layer_names = list(params_before.keys())
    print(f"找到 {len(layer_names)} 组参数进行比较。\n")

    for name in layer_names:
        if name not in params_after:
            print(f"警告：在'model_after'中找不到参数'{name}'，跳过。")
            continue

        # 提取对应层的参数，并转换为一维Numpy数组
        params1 = params_before[name].detach().cpu().numpy().flatten()
        params2 = params_after[name].detach().cpu().numpy().flatten()

        # 如果参数没有变化（例如，被冻结），JS散度应为0
        if np.array_equal(params1, params2):
            js_divergences[name] = 0.0
            continue

        # --- 步骤1: 确定共同范围 ---
        combined_min = min(params1.min(), params2.min())
        combined_max = max(params1.max(), params2.max())

        # 避免范围为0（例如该层所有参数都是同一个常数）
        if combined_min == combined_max:
            js_divergences[name] = 0.0
            continue

        # --- 步骤2: 构建概率分布 ---
        bins = np.linspace(combined_min, combined_max, num_bins + 1)
        hist1, _ = np.histogram(params1, bins=bins, density=True)
        hist2, _ = np.histogram(params2, bins=bins, density=True)

        # --- 步骤3: 平滑处理 ---
        epsilon = 1e-10
        hist1_smooth = hist1 + epsilon
        hist2_smooth = hist2 + epsilon

        prob_dist1 = hist1_smooth / np.sum(hist1_smooth)
        prob_dist2 = hist2_smooth / np.sum(hist2_smooth)

        # --- 步骤4: 计算JS散度 ---
        js_distance = jensenshannon(prob_dist1, prob_dist2)
        js_divergence = js_distance ** 2

        js_divergences[name] = js_divergence

    return js_divergences


def _get_block_priority(raw_js_dict):
    """
    将原始 JS 散度聚合为 Block 级别，并排序生成优先级列表。
    关键逻辑：过滤 Bias 和 BN，只看 Weight。
    """
    layer_groups = defaultdict(list)

    for param_name, js_value in raw_js_dict.items():
        # --- 1. 核心过滤逻辑 (只保留代表 "知识" 的 Weight) ---

        # 过滤掉 Bias (噪声源)
        if 'bias' in param_name: continue

        # 过滤掉 BN 层 (统计噪声)
        if 'bn' in param_name or 'downsample.1' in param_name or 'shortcut.1' in param_name:
            continue

        # --- 2. 智能分块 ---
        parts = param_name.split('.')

        # ResNet Block (e.g., layer1.0)
        if param_name.startswith('layer') and len(parts) >= 2:
            block_name = ".".join(parts[:2])

        # Linear/FC (e.g., linear, classifier.0)
        elif parts[0] in ['fc', 'linear', 'classifier']:
            # 如果是 classifier.0.weight，取 classifier.0
            if '.' in param_name:
                # 检查是否是类似 classifier.0 这样的结构
                if parts[1].isdigit():
                    block_name = ".".join(parts[:2])
                else:
                    block_name = parts[0]
            else:
                block_name = parts[0]

        # 其他 (e.g., conv1, features.0)
        else:
            if '.' in param_name:
                # features.0.weight -> features.0
                block_name = ".".join(parts[:-1])
            else:
                block_name = param_name

        layer_groups[block_name].append(js_value)

    # --- 3. 聚合 (取块内最大值) ---
    block_activity = {}
    for block_name, values in layer_groups.items():
        if values:
            block_activity[block_name] = max(values)

    # --- 4. 排序 (活跃度从高到低) ---
    # sorted 返回一个元组列表 [('layer3.2', 0.5), ('conv1', 0.1)...]
    sorted_blocks = sorted(block_activity.items(), key=lambda x: x[1], reverse=True)

    # 提取名字
    priority_list = [item[0] for item in sorted_blocks]

    return priority_list


def calculate_dynamic_priority(w_prev, w_curr):
    """
    主入口函数：输入两轮权重，输出排序后的层级列表。
    """
    print(">>> (Dynamic Layerwise) Calculating JS Divergence & Updating Priority...")
    raw_js = calculate_js_divergence_for_models(w_prev, w_curr)
    new_priority = _get_block_priority(raw_js)
    print(f">>> New Priority (Top 5): {new_priority[:5]}")
    return new_priority


def get_compressible_layers(model):
    """
    通用层级/块提取函数，适用于 CNN, VGG, ResNet。

    逻辑与 _group_params_by_block 保持一致：
    1. ResNet: 将同一个残差块内的所有卷积层视为一个整体 (Block)。
       例如: 'layer1.0.conv1.weight' 和 'layer1.0.conv2.weight' -> 归类为 'layer1.0'
    2. VGG/CNN: 每一层视为一个整体。
       例如: 'features.0.weight' -> 'features.0'
    3. 自动过滤: 剔除 BatchNorm 相关参数。

    Returns:
        block_names (list): 有序的 Block 名称列表 (无重复)。
                            HyperNet 的输出维度将等于 len(block_names)。
    """
    block_names = []
    seen_blocks = set()

    for name, _ in model.named_parameters():
        # --- 1. 黑名单过滤 (Filter) ---
        # 排除所有 BN 层的参数
        # (ResNet 特有: shortcut.1 是 BN, downsample.1 是 BN)
        # 只要名字里带 bn, running, tracked, 或者特定的 shortcut BN 后缀，就跳过
        if ('bn' in name) or ('shortcut.1' in name) or ('downsample.1' in name):
            continue

        # 也可以额外确保不包含 bias (如果你只想让 HyperNet 控制 weight)
        # 但为了保持 Block 名字的完整性，通常遍历到 weight 时记录即可
        # 这里我们遍历所有参数，通过去重机制(seen_blocks)来处理 weight 和 bias

        # --- 2. 智能分组逻辑 (Grouping) ---
        parts = name.split('.')

        # 逻辑 A: ResNet 的残差块 (e.g., layer1.0.conv1.weight)
        # 只要是以 'layer' 开头，且深度足够，就取前两部分作为 Block 名
        # 这样 layer1.0 下的所有 conv 都会被归到 'layer1.0' 这个组里
        if name.startswith('layer') and len(parts) >= 2:
            block_name = ".".join(parts[:2])  # 结果: layer1.0

        # 逻辑 B: 通用层 (VGG, CNN, ResNet的初始conv1和全连接linear)
        else:
            # VGG: 'features.0.weight' -> 'features.0'
            # CNN: 'conv1.weight' -> 'conv1'
            # Linear: 'linear.weight' -> 'linear'
            if '.' in name:
                # 去掉最后一部分 (即去掉 .weight 或 .bias)
                block_name = ".".join(parts[:-1])
            else:
                # 防御性代码: 只有参数名没有点的情况
                block_name = name

        # --- 3. 去重并保持顺序 ---
        # 我们只在第一次遇到这个 Block 时添加它
        if block_name not in seen_blocks:
            block_names.append(block_name)
            seen_blocks.add(block_name)

    return block_names

from typing import Any, Dict


def initialize_param_tracker(model: nn.Module, default_value: Any = 0.0) -> Dict[str, Any]:
    """
    根据模型形状，初始化一个跟踪字典。
    这个字典的键与 calculate_js_divergence_for_models 的输出键完全一致。

    Args:
        model (nn.Module): 你要分析的PyTorch模型。
        default_value (Any, optional): 你希望每个参数对应的初始值。
                                      例如：
                                      - 0.0 (用于初始化JS散度记录)
                                      - 0   (用于初始化 stable_counter)
                                      - False (用于初始化 is_frozen)
                                      默认为 0.0。

    Returns:
        dict: 一个字典，键是参数的名称 (如 'conv1.weight', 'conv1.bias'),
              值是 'default_value'。
    """

    # 1. model.named_parameters() 返回一个 (name, param) 的元组生成器
    # 2. dict(...) 将其转换为一个字典 {name: param_tensor, ...}
    # 3. .keys() 获取所有参数名称的列表
    param_names = dict(model.named_parameters()).keys()

    # 4. dict.fromkeys() 是最高效的方法：
    #    它从一个键的列表创建字典，并为所有键设置同一个默认值。
    tracker_dict = dict.fromkeys(param_names, default_value)

    print(f"成功初始化跟踪器，共找到 {len(tracker_dict)} 个参数。")
    return tracker_dict

def topK_exclude(record_w, cur_w, ratio, record_mask, record_mask2):
    common_keys = set(record_w.keys()).intersection(set(cur_w.keys()))

    w_delta = copy.deepcopy(cur_w)
    for key in common_keys:
        w_delta[key] = torch.abs(w_delta[key] - record_w[key])

    # available_indices = torch.where(mask_flat2 == 0)[0]
    mask_flat2 = torch.cat([record_mask2[k].flatten() for k in common_keys])

    w_delta_flat = torch.cat([v.flatten() for k, v in w_delta.items()])
    available_indices = torch.where(mask_flat2 == 0)[0]
    w_delta_available = w_delta_flat[available_indices]
    number_of_weights_to_filter = int(np.ceil(ratio * w_delta_flat.shape[0]))
    threshold = torch.sort(torch.abs(w_delta_available))[0][number_of_weights_to_filter]

    for key in common_keys:
        record_mask[key] = torch.where(torch.logical_and(w_delta[key] <= threshold, record_mask2[key]==0), 1, 0)
def topK_with_frozen(record_w, cur_w, ratio, record_mask, mask_is_frozen):
    w_delta = copy.deepcopy(cur_w)
    for key in w_delta.keys():
        # 仅考虑 mask 中为0的位置
        w_delta[key] = torch.abs(w_delta[key] - record_w[key])
        w_delta[key] *= (1 - mask_is_frozen[key])  # 仅保留 mask 中为0的位置

    w_delta_flat = torch.cat([v.flatten() for k, v in w_delta.items()])
    number_of_weights_to_filter = int(np.ceil(ratio * w_delta_flat.shape[0]))
    threshold = torch.sort(torch.abs(w_delta_flat))[0][number_of_weights_to_filter]

    for key in w_delta.keys():
        record_mask[key] = torch.where(w_delta[key] <= threshold, 1, 0)

def nextK(record_w, cur_w, ratio1, ratio2, record_mask):
    w_delta = copy.deepcopy(cur_w)
    for key in w_delta.keys():
        w_delta[key] = torch.abs(w_delta[key] - record_w[key])

    w_delta_flat = torch.cat([v.flatten() for k, v in w_delta.items()])
    number_of_weights_to_filter1 = int(np.ceil(ratio1 * w_delta_flat.shape[0]))
    threshold1 = torch.sort(torch.abs(w_delta_flat))[0][number_of_weights_to_filter1]
    number_of_weights_to_filter2 = int(np.ceil(ratio2 * w_delta_flat.shape[0]))
    threshold2 = torch.sort(torch.abs(w_delta_flat))[0][number_of_weights_to_filter2]

    for key in w_delta.keys():
        record_mask[key] = torch.where(torch.logical_or(w_delta[key] <= threshold1, w_delta[key] >= threshold2), 1, 0)

def bottomK(record_w, cur_w, ratio, record_mask):
    w_delta = copy.deepcopy(cur_w)
    for key in w_delta.keys():
        w_delta[key] = torch.abs(w_delta[key] - record_w[key])

    w_delta_flat = torch.cat([v.flatten() for k, v in w_delta.items()])
    number_of_weights_to_filter = int(np.ceil(ratio * w_delta_flat.shape[0]))
    threshold = torch.sort(torch.abs(w_delta_flat), descending=True)[0][number_of_weights_to_filter]

    for key in w_delta.keys():
        record_mask[key] = torch.where(w_delta[key] >= threshold, 1, 0)

def topK_new(record_w, cur_w, ratio, record_mask):
    w_delta = copy.deepcopy(cur_w)
    for key in w_delta.keys():
        w_delta[key] = torch.abs(w_delta[key] - record_w[key])

    w_delta_flat = torch.cat([v.flatten() for k, v in w_delta.items()])
    number_of_weights_to_filter = int(np.ceil(ratio * w_delta_flat.shape[0]))
    threshold = torch.sort(torch.abs(w_delta_flat), descending=True)[0][number_of_weights_to_filter]

    for key in w_delta.keys():
        record_mask[key] = torch.where(w_delta[key] >= threshold, 0, 1)

def calc_num_stable_params(mask):
    # 统计稳定的参数占比
    total_param = 0
    stable_param = 0
    for key in mask.keys():
        stable_param += torch.sum(mask[key]).int().item()
        total_param += mask[key].nelement()
    return round(stable_param / total_param, 4)

def random_mask(pct, w):
    # 随机覆盖 w 的 pct 的参数量
    # 先统计w的总的参数量，以及每层的参数范围，随机选择pct的参数量之后，对应下标都要减去开始下标
    # 最终获得每个 key 有对应的随机下标
    total_param = 0
    layer_be = []
    for key in w.keys():
        layer_be.append([total_param])
        total_param += w[key].nelement()
        layer_be[-1].append(total_param)
        layer_be[-1].append(key) #laybe的结构为[[0,t1,key1],[t1,t1+t2,key2]...]，即[[begin,end+1,key]...]
    rand_idxs = np.random.choice(range(total_param), int(pct * total_param), replace=False)
    rand_idxs = np.sort(rand_idxs)
    layer_idxs = {} #记录每个key，即每一层中选择的rand_idx的下标
    layer = 0
    for idx in rand_idxs:
        while idx >= layer_be[layer][1]:
            layer +=1
        key = layer_be[layer][2]
        begin = layer_be[layer][0]
        if key not in layer_idxs:
            layer_idxs[key] = []
        layer_idxs[key].append(idx - begin)

    return _random_mask(layer_idxs, w)

def _random_mask(layer_idxs, w):
    # 根据每一层获得的随机下标，得到mask掩码
    mask = {}
    for key in w.keys():
        if key not in layer_idxs:
            cur_mask = torch.zeros(w[key].nelement())
        else:
            idxs = layer_idxs[key]
            values = torch.ones(len(idxs))
            cur_mask = torch.zeros(w[key].nelement())
            cur_mask.scatter_(0, torch.tensor(idxs, dtype=torch.int64), values)
        mask[key] = cur_mask.view(w[key].shape).to(w[key].device)
    return mask


def recover_model_from_mask(w, record_w, record_mask):
    for key in w.keys():
        w[key] = torch.where(record_mask[key] == 1, record_w[key], w[key])  # 合并a,b两个tensor，如果a中元素大于0，则c中与a对应的位置取a的值，否则取b的值

def eval_avg_P_layers(E, E_abs):
    # 统计每层参数的，平均稳定指数
    p = copy.deepcopy(E)
    Ps = []
    preNum = 0
    for key in p.keys():
        p[key] = torch.abs(p[key]) / E_abs[key]
        p[key] = torch.where(torch.isnan(p[key]), torch.zeros_like(p[key]), p[key])
        b = torch.mean(p[key])
        # c = torch.sum(a)
        if "weight" in key:
            Ps.append(round(b.item(), 6))
            preNum = p[key].numel()
        elif "bias" in key:
            Ps[-1] = round((Ps[-1] * preNum + b.item() * p[key].numel()) / (preNum + p[key].numel()), 6)
    return Ps

# def and_mask(m1, m2):
#     mask = {}
#     for key in m1.keys():
#         mask[key] = m1[key] & m2[key]
#     return mask

def and_mask(m1, m2):
    mask = {}
    for key in m1.keys():
        mask[key] = torch.logical_and(m1[key].bool(), m2[key].bool()).float()
    return mask

# def or_mask(m1, m2):
#     mask = {}
#     for key in m1.keys():
#         mask[key] = m1[key] | m2[key]
#     return mask
def or_mask(m1, m2):
    mask = {}
    for key in m1.keys():
        # 将浮点数张量转换为布尔张量进行逻辑或运算
        mask[key] = torch.logical_or(m1[key].bool(), m2[key].bool()).float()
    return mask

def both_mask(m1, m2):
    mask1 = {}
    mask = {}
    for key in m1.keys():
        mask1[key] = torch.where((m1[key] == 1), 0, 1)

    for key in m2.keys():
        mask[key] = torch.where((m2[key] == 0), m2[key], mask1[key])
    return mask

def reverse_mask(m1):
    mask = {}
    for key in m1.keys():
        mask[key] = torch.where((m1[key] == 1), 0, 1)

    return mask




def gloab_mask(w_avg, w, ratio_global,mask_global):
    w_delta = copy.deepcopy(w)
    for key in w_delta.keys():
        w_delta[key] = torch.abs(w_delta[key] - w_avg[key])

    w_delta_flat = torch.cat([v.flatten() for k, v in w_delta.items()])
    number_of_weights_to_filter = int(np.ceil(ratio_global * w_delta_flat.shape[0]))
    threshold = torch.sort(torch.abs(w_delta_flat))[0][number_of_weights_to_filter]

    for key in w_delta.keys():
        mask_global[key] = torch.where(w_delta[key] >= threshold, 0, 1)

# def recover_model_from_mask(w, record_w, record_mask):
#     for key in w.keys():
#         w[key] = torch.where(record_mask[key] == 1, record_w[key], w[key])  # 合并a,b两个tensor，如果a中元素大于0，则c中与a对应的位置取a的值，否则取b的值

def recover_model_from_mask(w, record_w, record_mask):
    common_keys = set(w.keys()).intersection(set(record_w.keys()))

    for key in common_keys:
        w[key] = torch.where(record_mask[key] == 1, record_w[key], w[key])

def local_mask(w_avg, w, mask_global, mask_local,ratio_local): # w_avg和w分别为梯度下降之前和之后的权重
    w_delta = copy.deepcopy(w)
    w_delta1 = copy.deepcopy(w)
    w_delta2 = copy.deepcopy(w)
    for key in w_delta.keys(): # 获得delta
        w_delta[key] = torch.abs(w_delta[key] - w_avg[key])

    for key in w_delta1.keys(): #通过mask_global获得稀疏化后的delta
        w_delta1[key] = torch.where(mask_global[key] == 1, w_delta[key], 0)

    for key in w_delta2.keys(): #获得delta
        w_delta2[key] = torch.abs(w_delta[key] - w_delta1[key])

# def calculate_sum(w1, w2):
#     for key in w1.keys():
#         w1[key] = w1[key] + w2[key]
#     return w1

# def calculate_sum(w1, w2):
#     result = {}
#     for key in w1.keys():
#         result[key] = w1[key] + w2[key]
#     return result

# def calculate_sum(w1, w2):
#     result = {}
#     all_keys = set(w1.keys()).union(set(w2.keys()))
#     for key in all_keys:
#         # 如果 w1 和 w2 中都有对应键，则进行相加
#         if key in w1 and key in w2:
#             result[key] = w1[key] + w2[key]
#         elif key in w1:
#             # 如果只有 w1 中有对应键，则直接复制 w1 的值
#             result[key] = w1[key].clone()
#         elif key in w2:
#             # 如果只有 w2 中有对应键，则直接复制 w2 的值
#             result[key] = w2[key].clone()
#     return result

def calculate_sum(w1, w2):
    result = {}

    for key in set(w1.keys()).intersection(set(w2.keys())):
        result[key] = w1[key] + w2[key]

    return result

def calculate_sum_with_mask(w1, w2, mask):
    result = {}
    for key in w1.keys():
        result[key] = torch.where(mask[key] == 0, w1[key] + w2[key], w1[key])
    return result

# def calculate_error(w1, w2):
#     error_dict = {}
#     for key in w1.keys():
#         error_dict[key] = w1[key] - w2[key]
#     return error_dict

def calculate_error(w1, w2):
    error_dict = {}
    common_keys = set(w1.keys()).intersection(set(w2.keys()))
    for key in common_keys:
        error_dict[key] = w1[key] - w2[key]
    return error_dict

def calculate_upload_interval(base_size, speed_comm, upload_ratio, t_train, local_iter):
    # upload_ratio是一个浮点数
    per_comm = base_size * 4 / 1024 / 1024  # 单个模型的字节数 MB
    bitmap_comm = per_comm / 32  # 用于传输位置的字节数
    t_comm_mid = (per_comm * upload_ratio + bitmap_comm) / speed_comm  # 中间传输需要的时间
    iter = local_iter - int(t_comm_mid / t_train * local_iter)
    return iter

def calculate_multiple_upload_intervals(base_size, speed_comm, upload_ratio, t_train, local_iter):
    # upload_ratio是一个列表
    per_comm = base_size * 4 / 1024 / 1024  # 单个模型的字节数 MB
    bitmap_comm = per_comm / 32  # 用于传输位置的字节数
    t_comm = []
    for ratio in upload_ratio:
        t_comm.append((per_comm * ratio + bitmap_comm) / speed_comm)
    # print(t_comm)
    iters = []
    upload_interval = local_iter
    reversed_t_comm = t_comm[::-1]
    # print(reversed_t_comm)
    for t in reversed_t_comm:
        upload_interval = upload_interval - int(t / t_train * local_iter)
        iters.append(upload_interval)
    reversed_iters = iters[::-1]
    return reversed_iters

def calculate_receive_powers(radius, device_num, transmit_power_pi, shadow_fading_standard_deviation):
    print("Radius(Km):", radius)
    print("Device_num:", device_num)
    print("Transmit_power(dB):", transmit_power_pi)
    # 计算每台设备的path_loss
    device_distances = np.random.uniform(low=0.01, high=radius, size=device_num)
    print(f"Device_distances(km):", device_distances)
    path_losses = np.array([128.1 + 37.6 * np.log10(d) for d in device_distances])
    print(f"Path_losses(dB):", path_losses)

    # 计算每台设备的shadow_fading
    # 生成1000个长度为10的阴影衰落数组
    shadow_fading_arrays = []
    for _ in range(1000):
        shadow_fading = np.random.normal(0, shadow_fading_standard_deviation, device_num)
        shadow_fading_arrays.append(shadow_fading)
    # 将列表转换为 NumPy 数组
    # shadow_fading_arrays = np.array(shadow_fading_arrays)
    # print(shadow_fading_arrays)
    # 计算每列的均值
    shadow_fadings = np.mean(shadow_fading_arrays, axis=0)
    print(f"Shadow_fadings(dB):",shadow_fadings)

    # 计算每台设备的接收功率
    receive_powers = transmit_power_pi - path_losses - shadow_fadings

    return receive_powers

def equation(bi, upload_k, N0, tau_d, tau_i_train, receive_power, d):
    return bi * (tau_d - tau_i_train) * np.log2(1 + (10**((receive_power + 30 - 30) / 10)) / (bi * (10**((N0 - 30) / 10)))) - (32 * ((1 - upload_k / d) + 1) * d )
    # return bi * (tau_d - tau_i_train) * np.log2(1 + (10 ** ((receive_power + 30 - 30) / 10)) / (bi * (10 ** ((N0 - 30) / 10)))) - (32 * ((1 - upload_k / d) + 1) * d)

def solve_equation_bisection(upload_k, N0, tau_d, tau_i_train, receive_power, d, B = 20000000, tolerance=1e-8, max_iterations=1000):
    # 设置初始搜索范围
    lower_bound = 0.0
    upper_bound = B

    for iteration in range(max_iterations):
        bi_guess = (lower_bound + upper_bound) / 2
        equation_value = equation(bi_guess, upload_k, N0, tau_d, tau_i_train, receive_power, d)
        # equation_value = bi_guess * (tau_d - tau_i_train) * np.log2(1 + (10**((receive_power + 30 - 30) / 10)) / (bi_guess * (10**((N0 - 30) / 10)))) - (32 * ((1 - upload_k / d) + 1) * d )
        # print(equation_value)


        if np.abs(equation_value) < tolerance:
            # print("Successfully")
            print(iteration,end = ',')
            return bi_guess

        if equation_value > 0:
            upper_bound = bi_guess
        else:
            lower_bound = bi_guess

def replace_values(w1, w2):
    for key in w1.keys():
        if key in w2:
            w1[key] = w2[key]


def print_state_dict_keys(state_dict, title="检查 state_dict"):
    """
    一个辅助函数，用于打印 state_dict 中的键，并对它们进行分类，
    特别关注 BatchNorm 的缓冲区是否丢失。
    """

    print("\n" + "=" * 50)
    print(f"--- {title} ---")

    # 检查输入是否为字典
    if not isinstance(state_dict, dict):
        print(f"错误: 输入不是一个字典 (dict), 而是 {type(state_dict)}")
        print("=" * 50)
        return

    all_keys = list(state_dict.keys())
    total_keys = len(all_keys)

    if total_keys == 0:
        print("字典为空 (0 keys).")
        print("=" * 50)
        return

    # 分类键
    param_keys = []
    buffer_keys = []
    other_keys = []

    for key in all_keys:
        if 'weight' in key or 'bias' in key:
            param_keys.append(key)
        elif 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            buffer_keys.append(key)
        else:
            other_keys.append(key)

    print(f"总键数 (Total Keys): {total_keys}")
    print(f"  - 参数 (Parameters, e.g., weight/bias): {len(param_keys)}")
    print(f"  - 缓冲区 (Buffers, e.g., running_mean): {len(buffer_keys)}")
    if other_keys:
        print(f"  - 其他 (Others): {len(other_keys)}")

    # --- 关键的诊断信息 ---
    if len(buffer_keys) == 0 and len(param_keys) > 0:
        print("\n!!! 警告: 字典中包含参数, 但 *没有* 缓冲区 (Buffers) !!!")
        print("!!! 这极有可能是导致 'Missing key(s)' 错误的原因 !!!")

    elif len(buffer_keys) > 0:
        print(f"\n找到的缓冲区示例 (Example Buffers found):")
        # 只打印前3个，避免刷屏
        for i, key in enumerate(buffer_keys):
            if i < 3:
                print(f"  - {key}")
            else:
                print(f"  - ... (以及其他 {len(buffer_keys) - 3} 个)")
                break

    print("=" * 50)


def inspect_model_keys(model):
    """
    打印模型 state_dict 中每个键的名称、形状和参数总量。
    并在最后统计总参数量。
    """
    print("\n" + "=" * 85)
    print(f"{'Key Name (键名)':<45} | {'Shape (形状)':<20} | {'Count (数量)':<15}")
    print("-" * 85)

    total_params = 0
    total_buffers = 0

    # 获取 state_dict 以包含所有参数和缓冲区
    state_dict = model.state_dict()

    for key, tensor in state_dict.items():
        # 计算该张量的元素总数
        count = tensor.numel()
        shape = str(list(tensor.shape))

        # 简单的分类统计 (用于最后总结)
        if 'running' in key or 'tracked' in key:
            total_buffers += count
        else:
            total_params += count

        # 打印每一行
        # 使用 comma (,) 格式化数字，使其更易读 (例如 1,000,000)
        print(f"{key:<45} | {shape:<20} | {count:<15,}")

    total_elements = total_params + total_buffers
    print("-" * 85)
    print(f"Total Parameters (Weight/Bias): {total_params:,}")
    print(f"Total Buffers (Mean/Var):       {total_buffers:,}")
    print(f"Total Elements:                 {total_params + total_buffers:,}")
    print("=" * 85 + "\n")

    return total_params, total_buffers, total_elements

if __name__=="__main__":
    # m1 = cnn.CNN()
    # # m2 = cnn.CNN()
    # w1 = m1.state_dict()
    # # w2 = m2.state_dict()
    # # diff = diff_model(w1, w2)
    # # print(diff)
    # # cloned_tensor = get_model_like_tensor(m1)
    # mask = random_mask(0.3, w1)
    # pct = calc_num_stable_params(mask)
    # print()
    #
    # c = 0
    # print(int(c))
    #
    # print((122570 * 4 / 1024 / 1024 * 0.5 + 122570 * 4 / 1024 / 1024 / 32) / 0.07)
    # print((122570 * 4 / 1024 / 1024 * 0.5) / 0.07)
    # print((122570 * 4 / 1024 / 1024 * 0.3 + 122570 * 4 / 1024 / 1024 / 32) / 0.066)
    # print(calculate_multiple_upload_intervals(122570, 0.062, [0.1, 0.1, 0.1], 4.365, 250))
    # print(calculate_multiple_upload_intervals(122570, 0.062, [0.15, 0.15], 4.365, 250))
    #
    # print(len([0.8415, 0.8414, 0.8413, 0.8413, 0.8414, 0.8412, 0.8415, 0.8414, 0.8413, 0.8413, 0.8413, 0.8413, 0.8413, 0.8413, 0.8412, 0.8413, 0.8413, 0.8414, 0.8414, 0.8414, 0.8412, 0.8412, 0.8413, 0.8413, 0.8412, 0.8412, 0.8412, 0.8413, 0.8413, 0.8412, 0.8411, 0.8412, 0.8412, 0.8411, 0.8412, 0.8411, 0.8412, 0.8412, 0.8412, 0.8412, 0.8411, 0.8411, 0.8411, 0.8411, 0.8411, 0.8412, 0.8412, 0.8411, 0.8411, 0.8412, 0.8412, 0.8411, 0.8412, 0.8411, 0.8411, 0.8412, 0.8412, 0.8412, 0.8412, 0.8413, 0.8412, 0.8412, 0.8412, 0.8412, 0.8412, 0.8412, 0.8412, 0.8413, 0.8412, 0.8412, 0.8414, 0.8414, 0.8414, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8414, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8415, 0.8415, 0.8414, 0.8415, 0.8415, 0.8415, 0.8414, 0.8415, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8414, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8415, 0.8414, 0.8414, 0.8414, 0.8415, 0.8416, 0.8416, 0.8416, 0.8417, 0.8417, 0.8417, 0.8417, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8417, 0.8416, 0.8416, 0.8415, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8417, 0.8417, 0.8417, 0.8417, 0.8416, 0.8417, 0.8416, 0.8416, 0.8417, 0.8416, 0.8416, 0.8417, 0.8416, 0.8416, 0.8416, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8416, 0.8416, 0.8416, 0.8417, 0.8417, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8417, 0.8417, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8416, 0.8417, 0.8417, 0.8416, 0.8416, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417, 0.8417]))
    # print(len([0.8445, 0.8444, 0.8445, 0.8445, 0.8445, 0.8444, 0.8445, 0.8444, 0.8445, 0.8444, 0.8443, 0.8444, 0.8445, 0.8443, 0.8443, 0.8443, 0.8443, 0.8444, 0.8444, 0.8444, 0.8443, 0.8443, 0.8444, 0.8444, 0.8443, 0.8443, 0.8444, 0.8443, 0.8444, 0.8444, 0.8444, 0.8444, 0.8444, 0.8444, 0.8444, 0.8444, 0.8444, 0.8444, 0.8444, 0.8443, 0.8443, 0.8443, 0.8441, 0.8441, 0.8443, 0.8443, 0.8442, 0.8443, 0.8443, 0.8444, 0.8443, 0.8443, 0.8443, 0.8443, 0.8444, 0.8444, 0.8444, 0.8443, 0.8441, 0.8443, 0.8443, 0.8442, 0.8443, 0.8443, 0.8443, 0.8444, 0.8444, 0.8444, 0.8444, 0.8444, 0.8443, 0.8443, 0.8444, 0.8444, 0.8444, 0.8444, 0.8443, 0.8443, 0.8443, 0.8443, 0.8443, 0.8443, 0.8443, 0.8444, 0.8443, 0.8444, 0.8443, 0.8444, 0.8443, 0.8443, 0.8443, 0.8443, 0.8443, 0.8443, 0.8443, 0.8442, 0.844, 0.844, 0.8441, 0.8441, 0.8441, 0.8441, 0.844, 0.844, 0.844, 0.844, 0.844, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8442, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.844, 0.8442, 0.8441, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8443, 0.8443, 0.8443, 0.8443, 0.8443, 0.8443, 0.8443, 0.8443, 0.8443, 0.8442, 0.8443, 0.8443, 0.8443, 0.8443, 0.8443, 0.8443, 0.8443, 0.8442, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8443, 0.8443, 0.8443, 0.8443, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8442, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.8441, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844]))
    # print(len([0.0809, 0.145, 0.1599, 0.1661, 0.1996, 0.2165, 0.2552, 0.2627, 0.2824, 0.3081, 0.3358, 0.3495, 0.3433, 0.3785, 0.3671, 0.3926, 0.411, 0.4192, 0.4272, 0.4437, 0.4563, 0.4603, 0.4755, 0.4859, 0.4902, 0.4989, 0.5057, 0.5145, 0.5276, 0.5306, 0.5412, 0.5449, 0.5523, 0.5702, 0.5754, 0.5892, 0.6019, 0.5995, 0.6116, 0.6275, 0.6235, 0.63, 0.6397, 0.6424, 0.6516, 0.657, 0.6581, 0.663, 0.667, 0.6691, 0.6764, 0.6784, 0.6893, 0.6934, 0.7037, 0.7044, 0.7082, 0.7126, 0.713, 0.7189, 0.721, 0.7219, 0.7262, 0.7299, 0.7331, 0.7351, 0.7378, 0.7424, 0.7468, 0.7455, 0.747, 0.7488, 0.751, 0.7508, 0.7512, 0.757, 0.7572, 0.7596, 0.7587, 0.7608, 0.7604, 0.7648, 0.7638, 0.7678, 0.7717, 0.7757, 0.7757, 0.7751, 0.7787, 0.7784, 0.7802, 0.7839, 0.7817, 0.7819, 0.7815, 0.7806, 0.7833, 0.7858, 0.7842, 0.785, 0.7902, 0.7885, 0.7919, 0.7909, 0.7929, 0.7932, 0.7919, 0.7968, 0.7963, 0.7958, 0.8001, 0.7976, 0.8034, 0.798, 0.8003, 0.7999, 0.7994, 0.8012, 0.8016, 0.7992, 0.8023, 0.8027, 0.804, 0.8054, 0.8101, 0.8101, 0.8105, 0.811, 0.8098, 0.8084, 0.8087, 0.8113, 0.8095, 0.8086, 0.8137, 0.8139, 0.8139, 0.8164, 0.816, 0.8166, 0.8155, 0.8144, 0.816, 0.8162, 0.8198, 0.8187, 0.8179, 0.8159, 0.8184, 0.821, 0.8185, 0.8196, 0.8209, 0.8222, 0.8198, 0.8187, 0.8223, 0.8226, 0.8219, 0.8235, 0.8231, 0.8228, 0.8207, 0.8232, 0.8247, 0.8251, 0.8251, 0.8258, 0.8253, 0.8248, 0.8234, 0.8241, 0.8239, 0.8247, 0.8251, 0.8254, 0.8278, 0.8256, 0.8263, 0.8267, 0.829, 0.8274, 0.8264, 0.8293, 0.827, 0.8257, 0.8281, 0.831, 0.8292, 0.8282, 0.8286, 0.8275, 0.8288, 0.8279, 0.8295, 0.828, 0.8301, 0.8291, 0.8327, 0.8312, 0.8321, 0.8318, 0.831, 0.8292, 0.83, 0.8291, 0.8319, 0.8302, 0.8318, 0.8317, 0.832, 0.8325, 0.83, 0.8316, 0.8303, 0.8296, 0.8315, 0.8312, 0.8331, 0.8326, 0.8337, 0.8318, 0.8329, 0.831, 0.8322, 0.8313, 0.8339, 0.8321, 0.8328, 0.8326, 0.8331, 0.8339, 0.8327, 0.8339, 0.8344, 0.8339, 0.8329, 0.8355, 0.835, 0.8343, 0.8324, 0.8346, 0.836, 0.8345, 0.8345, 0.8334, 0.8344, 0.8358, 0.8366, 0.8353, 0.8347, 0.8346, 0.8348, 0.8368, 0.8375, 0.8362, 0.8366, 0.8361, 0.8364, 0.837, 0.8371, 0.8364, 0.8367, 0.838, 0.8384, 0.8365, 0.8357, 0.8353, 0.8364, 0.836, 0.8367, 0.834, 0.8381, 0.8366, 0.8376, 0.8373, 0.8377, 0.8354, 0.8362, 0.8386, 0.8362, 0.8365, 0.8362, 0.8376, 0.8384, 0.838, 0.8359, 0.8383, 0.8391, 0.8389, 0.8386, 0.8379, 0.8377, 0.8368, 0.8376, 0.8389, 0.8383, 0.8385, 0.838, 0.8385, 0.8391, 0.8393, 0.8387, 0.8389, 0.838, 0.8379, 0.8394, 0.8403, 0.84, 0.8398, 0.8384, 0.8394, 0.8383, 0.8386, 0.839, 0.8392, 0.8391, 0.8402, 0.8388, 0.8385, 0.8397, 0.8395, 0.8395, 0.8394, 0.839, 0.8404, 0.8396, 0.8389, 0.8395, 0.8397, 0.8388, 0.8375, 0.8389, 0.8403, 0.8401, 0.8405, 0.8405, 0.8383, 0.8381, 0.8384, 0.839, 0.8402, 0.8393, 0.8393, 0.8395, 0.8405, 0.8385, 0.8394, 0.8382, 0.8381, 0.8401, 0.8402, 0.839, 0.8393, 0.8385, 0.8376, 0.8383, 0.8385, 0.8378, 0.8389, 0.8398, 0.8381, 0.8383, 0.8397, 0.84, 0.8398, 0.8403, 0.8395, 0.8389, 0.8395, 0.8389, 0.8389, 0.8396, 0.8391, 0.8399, 0.8399, 0.8387, 0.8373, 0.8389, 0.8377, 0.8385, 0.8388, 0.84, 0.8391, 0.839, 0.8398, 0.8385, 0.8385, 0.8389, 0.838, 0.8386, 0.8391, 0.8396, 0.8405, 0.8395, 0.8394, 0.8396, 0.8402, 0.8399, 0.8393, 0.8395, 0.8393, 0.8398, 0.8406, 0.8411, 0.8412, 0.8404, 0.8409, 0.8409, 0.8408, 0.8406, 0.8405, 0.8395, 0.8404, 0.8403, 0.8396, 0.8401, 0.8407, 0.8399, 0.84, 0.8408, 0.8408, 0.8407, 0.8406, 0.8408, 0.8404, 0.8412, 0.8407, 0.84, 0.8399, 0.8399, 0.8399, 0.8395, 0.8397, 0.8399, 0.8395, 0.84, 0.8392, 0.8403, 0.8404, 0.8393, 0.839, 0.8405, 0.8404, 0.8395, 0.8401, 0.8407, 0.8404, 0.8395, 0.8402, 0.8412, 0.8404, 0.8404, 0.8401, 0.8407, 0.8409, 0.8409, 0.8409, 0.8407, 0.8408, 0.8411, 0.8409, 0.8405, 0.8406, 0.8411, 0.841, 0.8407, 0.8402, 0.8406, 0.8403, 0.8403, 0.8394, 0.8403, 0.8399, 0.8403, 0.8404, 0.8411, 0.8406, 0.8411, 0.8412, 0.8411, 0.8408, 0.841, 0.8406, 0.8413, 0.841, 0.8415, 0.8409, 0.8416, 0.8399, 0.8407, 0.8416, 0.8406, 0.8406, ]))

    from torchvision.models import vgg11, resnet18


    # 简单的 CNN 模拟
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(6 * 14 * 14, 10)


    # --- 测试 CNN ---
    print("\n>>> 1. Testing SimpleCNN:")
    model_cnn = SimpleCNN()
    blocks = get_compressible_layers(model_cnn)
    print(f"Blocks ({len(blocks)}): {blocks}")
    # 预期: ['conv1', 'fc1']

    # --- 测试 VGG ---
    print("\n>>> 2. Testing VGG11:")
    model_vgg = vgg11()
    blocks = get_compressible_layers(model_vgg)
    # VGG 的 features 是 Sequential，包含 conv(0), relu(1), pool(2), conv(3)...
    print(f"Blocks ({len(blocks)}): {blocks[:5]} ...")
    # 预期: ['features.0', 'features.3', 'features.6', ...]

    # --- 测试 ResNet ---
    print("\n>>> 3. Testing ResNet18:")
    model_res = resnet18()
    blocks = get_compressible_layers(model_res)
    print(f"Blocks ({len(blocks)}): {blocks}")