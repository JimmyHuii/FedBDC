import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from collections import defaultdict
import numpy as np

# [关键新增] 引入 functional_call 用于无副作用的前向传播
try:
    from torch.func import functional_call
except ImportError:
    # 兼容旧版本 PyTorch
    from torch.nn.utils.stateless import functional_call


# ==============================================================================
# 1. HyperNetwork 模型定义
# ==============================================================================
class E_HyperNet(nn.Module):
    def __init__(self, num_layers, embed_dim=32, hidden_dim=64):
        super(E_HyperNet, self).__init__()
        self.embedding = nn.Embedding(num_layers, embed_dim)

        # 输入: Embedding + 3个物理特征 (log_N, C_comm, C_comp)
        input_dim = embed_dim + 3
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.head_down = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.head_up = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, layer_indices, physical_stats):
        embeds = self.embedding(layer_indices)
        x = torch.cat([embeds, physical_stats], dim=1)
        features = self.shared_mlp(x)
        alpha_down = self.head_down(features).squeeze()
        alpha_up = self.head_up(features).squeeze()
        return alpha_down, alpha_up


# ==============================================================================
# 2. 核心工具函数
# ==============================================================================
def ste_soft_mask(weights, alpha):
    """直通估计器 (STE): 前向硬截断，反向软梯度"""
    k = int(alpha.item() * weights.numel())
    k = max(k, 1)
    # view(-1) 展平, kthvalue 找第 k 小
    threshold = weights.abs().view(-1).kthvalue(weights.numel() - k + 1).values
    mask_hard = (weights.abs() >= threshold).float()

    # 核心：反向传播时欺骗梯度，让 mask 的梯度等于 alpha 的梯度
    mask_ste = (mask_hard - alpha).detach() + alpha
    return mask_ste


def get_model_blocks(model):
    """提取 Block 名称列表"""
    block_names = []
    seen_blocks = set()
    for name, _ in model.named_parameters():
        if ('bn' in name) or ('downsample.1' in name) or ('shortcut.1' in name):
            continue
        parts = name.split('.')
        if name.startswith('layer') and len(parts) >= 2:
            block_name = ".".join(parts[:2])
        elif '.' in name:
            block_name = ".".join(parts[:-1])
        else:
            block_name = name
        if block_name not in seen_blocks:
            block_names.append(block_name)
            seen_blocks.add(block_name)
    return block_names


def get_block_physical_stats(model, block_names, pi_up, pi_down, up_rate_list, down_rate_list, energy_per_flop_list,
                             device):
    """计算每个 Block 的聚合物理特征"""
    # 1. 聚合每个 Block 的 Params 和 FLOPs
    block_data = defaultdict(lambda: {'params': 0, 'flops': 0})
    for name, param in model.named_parameters():
        parts = name.split('.')
        if name.startswith('layer') and len(parts) >= 2:
            b_name = ".".join(parts[:2])
        elif '.' in name:
            b_name = ".".join(parts[:-1])
        else:
            b_name = name

        if b_name in block_names and 'weight' in name:
            p_count = param.numel()
            block_data[b_name]['params'] += p_count
            reuse = 100 if ('conv' in name or 'features' in name) else 1
            block_data[b_name]['flops'] += p_count * reuse

    # 2. 计算平均通信成本
    R_up = np.maximum(np.array(up_rate_list), 1e-6)
    R_down = np.maximum(np.array(down_rate_list), 1e-6)
    costs_i = (pi_up / R_up) + (pi_down / R_down)
    avg_comm_cost_per_bit = np.mean(costs_i)
    c_comm_val = avg_comm_cost_per_bit * 32

    # 3. 平均计算成本
    avg_energy_per_flop = np.mean(np.array(energy_per_flop_list))

    # 4. 生成特征向量
    raw_list = []
    for b_name in block_names:
        N_block = block_data[b_name]['params']
        Flops_block = block_data[b_name]['flops']
        c_comp = (Flops_block * 3 * avg_energy_per_flop) / (N_block + 1e-6)
        raw_list.append([N_block, c_comm_val, c_comp])

    # 5. 归一化
    if not raw_list: return torch.zeros(0, 3).to(device)
    raw_tensor = torch.tensor(raw_list, dtype=torch.float32).to(device)

    log_N = torch.log10(raw_tensor[:, 0] + 1)
    mean = raw_tensor.mean(dim=0)
    std = raw_tensor.std(dim=0) + 1e-6
    if std[1] < 1e-9:
        c_comm_norm = torch.zeros_like(raw_tensor[:, 1])
    else:
        c_comm_norm = (raw_tensor[:, 1] - mean[1]) / std[1]
    c_comp_norm = (raw_tensor[:, 2] - mean[2]) / std[2]

    return torch.stack([log_N, c_comm_norm, c_comp_norm], dim=1)


# ==============================================================================
# 3. 封装好的主调用函数
# ==============================================================================
class HyperNetController:
    def __init__(self, args, global_model, device):
        self.args = args
        self.model = global_model
        self.device = device
        self.block_names = get_model_blocks(global_model)
        self.num_blocks = len(self.block_names)
        self.block_param_counts = self._count_block_params()

        print(f"[HyperNet] Managing {self.num_blocks} blocks.")

        self.net = E_HyperNet(self.num_blocks).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def _count_block_params(self):
        counts = defaultdict(int)
        for name, param in self.model.named_parameters():
            parts = name.split('.')
            if name.startswith('layer') and len(parts) >= 2:
                b_name = ".".join(parts[:2])
            elif '.' in name:
                b_name = ".".join(parts[:-1])
            else:
                b_name = name
            if b_name in self.block_names and 'weight' in name:
                counts[b_name] += param.numel()
        return torch.tensor([counts[b] for b in self.block_names], dtype=torch.float32).to(self.device)

    def run_optimization_step(self, proxy_data, proxy_target, pi_up, pi_down, up_rate_list, down_rate_list,
                              energy_per_flop_list):
        """
        执行一次 HyperNet 的更新并返回下一轮的策略
        """
        self.net.train()
        self.optimizer.zero_grad()

        # A. 准备输入
        stats = get_block_physical_stats(
            self.model, self.block_names,
            pi_up, pi_down,
            up_rate_list, down_rate_list,
            energy_per_flop_list,
            self.device
        )
        block_ids = torch.arange(self.num_blocks).to(self.device)

        # B. 前向传播
        alpha_down, alpha_up = self.net(block_ids, stats)

        # C. 模拟训练过程 (STE + Functional Call) [修复错误的关键部分]
        alpha_map = {name: val for name, val in zip(self.block_names, alpha_down)}

        # 1. 构建一个包含"软参数"的字典
        soft_params = {}

        # 为了兼容 functional_call，我们需要把 buffers 也放进去
        for name, buf in self.model.named_buffers():
            soft_params[name] = buf

        for name, param in self.model.named_parameters():
            parts = name.split('.')
            b_name = ".".join(parts[:2]) if name.startswith('layer') and len(parts) >= 2 else (
                ".".join(parts[:-1]) if '.' in name else name)

            if b_name in alpha_map and 'weight' in name:
                # 使用 STE 生成软掩码
                mask = ste_soft_mask(param, alpha_map[b_name])
                # [核心修复] 这里我们不修改 param.data，而是生成一个新的 Tensor
                # 这个新的 Tensor 依然在计算图中，且没有副作用
                soft_params[name] = param * mask
            else:
                soft_params[name] = param  # 不需要压缩的参数直接引用

        # 2. 使用 functional_call 进行无副作用的前向传播
        # 这会让模型使用 soft_params 中的权重进行计算，而不修改 self.model 本身
        output = functional_call(self.model, soft_params, proxy_data)

        task_loss = F.cross_entropy(output, proxy_target)

        # D. 计算 Energy Loss
        # energy_loss = torch.mean(alpha_down * stats[:, 1] + alpha_up * stats[:, 1])
        #
        # lambda_reg = self.args.lambda_reg
        # total_loss = task_loss + lambda_reg * energy_loss
        # ==================== D. 计算 Energy Loss ====================
        energy_penalties = []

        # 你的 self.block_names 是一个 List，所以直接 enumerate 即可
        # 取出来的是 index (block_id) 和 字符串 (name)
        for block_id, name in enumerate(self.block_names):

            # 1. 基础的单层能量消耗 (通信代价)
            layer_energy = alpha_down[block_id] * stats[block_id, 1] + alpha_up[block_id] * stats[block_id, 1]

            # 2. [核心修改]: 针对 AlexNet 的全连接层 (classifier) 施加惩罚
            if hasattr(self.args, 'model') and self.args.model == 'alexnet' and 'classifier' in name:
                # 施加 50 倍的超级惩罚！
                layer_energy = layer_energy * self.args.rate

            energy_penalties.append(layer_energy)

        # 3. 将所有层的惩罚组合成一个 Tensor 并求平均
        energy_loss = torch.mean(torch.stack(energy_penalties))

        lambda_reg = self.args.lambda_reg
        total_loss = task_loss + lambda_reg * energy_loss

        # E. 反向传播
        total_loss.backward()
        self.optimizer.step()

        # F. 生成最终策略
        self.net.eval()
        with torch.no_grad():
            final_alpha_down, final_alpha_up = self.net(block_ids, stats)

            # 计算总参数量
            k_down_list = torch.floor(final_alpha_down * self.block_param_counts)
            k_up_list = torch.floor(final_alpha_up * self.block_param_counts)
            k_down_total = torch.sum(k_down_list).item()
            k_up_total = torch.sum(k_up_list).item()

            # [关键修改] 将 Tensor 转换为 Block Name -> Alpha Value 的字典
            # 这样 topK_hypernet 可以直接使用
            alpha_down_dict = {name: val.item() for name, val in zip(self.block_names, final_alpha_down)}
            alpha_up_dict = {name: val.item() for name, val in zip(self.block_names, final_alpha_up)}

        return alpha_down_dict, alpha_up_dict, int(k_down_total), int(k_up_total), total_loss.item()