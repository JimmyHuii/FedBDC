import torch
import copy


def qsgd_quantize_tensor(tensor: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """
    [底层原子函数]
    只专注于数学逻辑：如何将一个浮点张量压缩为低比特表示并还原。
    它不关心这个张量是权重、梯度还是偏差，也不关心它属于哪一层。

    Args:
        tensor: 待量化的浮点数张量
        num_bits: 量化位数 (例如 4 或 8)。s = 2^bits - 1。

    Returns:
        quantized_tensor: 带有量化噪声的浮点数张量。
    """
    # 0. 展平张量以便统一处理 (Vector-wise QSGD)
    shape = tensor.shape
    t_flat = tensor.view(-1)

    # 1. 计算范数 (L2 Norm)
    norm = torch.norm(t_flat)

    # 边界保护：如果更新量为0，直接返回
    if norm <= 1e-9:
        return tensor

    # 2. 归一化 (Normalize) -> 取绝对值并除以范数，范围 [0, 1]
    abs_val = torch.abs(t_flat)
    sign = torch.sign(t_flat)
    normalized = abs_val / norm

    # 3. 计算量化等级 s
    s = 2 ** num_bits - 1

    # 4. 随机舍入 (Stochastic Rounding)
    # 将 [0, 1] 映射到 [0, s]
    scaled = normalized * s
    l = torch.floor(scaled)  # 下界整数
    prob = scaled - l  # 小数部分作为向上取整的概率

    # 伯努利采样：以 prob 的概率 +1 (向上取整)，否则 +0 (向下取整)
    # 这一步保证了 E[quantized] = original，即无偏估计
    delta = torch.bernoulli(prob)
    quantized_int = l + delta

    # 5. 反量化 (还原)
    # 还原公式: sign * (int / s) * norm
    restored = sign * (quantized_int / s) * norm

    return restored.view(shape)


def compute_and_quantize_update(local_model_dict, global_model_dict, num_bits=4):
    """
    [业务逻辑封装函数] - 旧版本，包含计算差值的逻辑
    """
    quantized_update = {}

    for key in local_model_dict.keys():
        raw_delta = local_model_dict[key] - global_model_dict[key]

        if 'weight' in key and raw_delta.numel() > 1:
            quantized_update[key] = qsgd_quantize_tensor(raw_delta, num_bits=num_bits)
        else:
            quantized_update[key] = raw_delta

    return quantized_update


def quantize_state_dict(target_dict, num_bits=4):
    """
    [通用量化函数] - 你想要的新函数
    直接对传入的 state_dict (可以是更新量 u，也可以是完整模型) 进行原位量化模拟。

    特点：
    1. 不执行减法，只负责量化。
    2. 自动过滤非浮点类型 (如 num_batches_tracked)。
    3. 只量化 weight，保留 bias 和 BN 统计量的高精度 (符合 FedPAQ 标准)。

    Args:
        target_dict (dict): 待量化的字典 (例如你的更新量 u)
        num_bits (int): 量化位数

    Returns:
        new_dict (dict): 量化并还原后的字典
    """
    new_dict = {}

    for key, tensor in target_dict.items():
        # 1. 安全检查：确保是 tensor 且是浮点数 (避免量化整数 buffer)
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            new_dict[key] = tensor
            continue

        # 2. 策略检查：只量化 'weight' 参数
        # Bias 和 BN 的 running_mean/var 通常不量化，因为体积小且敏感
        if 'weight' in key and tensor.numel() > 1:
            new_dict[key] = qsgd_quantize_tensor(tensor, num_bits=num_bits)
        else:
            # 保持原样 (Full Precision)
            new_dict[key] = tensor

    return new_dict