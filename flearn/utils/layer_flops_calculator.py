import torch
import torch.nn as nn
import numpy as np


class LayerFlopsProfiler:
    def __init__(self, model, input_size=(1, 3, 32, 32), device='cpu'):
        self.model = model
        self.input_size = input_size
        self.device = device
        self.layer_stats = []
        self.hooks = []

    def _count_conv2d(self, layer, input_shape, output_shape):
        # input_shape: (Batch, Cin, Hin, Win)
        # output_shape: (Batch, Cout, Hout, Wout)
        # Kernel: (Cout, Cin, K, K)

        batch_size = output_shape[0]
        out_h, out_w = output_shape[2], output_shape[3]

        # Kernel ops = Cin * K * K
        kernel_ops = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]

        # MACs per output pixel = Kernel ops
        # Total MACs = Batch * Cout * Hout * Wout * Kernel ops
        total_macs = batch_size * layer.out_channels * out_h * out_w * kernel_ops

        # Bias adds one FLOP per output pixel
        if layer.bias is not None:
            total_macs += batch_size * layer.out_channels * out_h * out_w

        # 1 MAC approx 2 FLOPs (Multiply + Add)
        return total_macs * 2

    def _count_linear(self, layer, input_shape, output_shape):
        # input_shape: (Batch, ..., In_Features)
        # output_shape: (Batch, ..., Out_Features)

        # 展平 batch 维度以外的所有维度计算总 token 数
        # 例如 input (1, 10) -> total_tokens = 1
        total_tokens = 1
        for s in output_shape[:-1]:
            total_tokens *= s

        # MACs = Tokens * In * Out
        total_macs = total_tokens * layer.in_features * layer.out_features

        if layer.bias is not None:
            total_macs += total_tokens * layer.out_features

        return total_macs * 2

    def _hook_fn(self, layer, inputs, outputs):
        # 获取输入输出形状
        input_shape = inputs[0].shape
        output_shape = outputs.shape

        flops = 0
        params = 0

        # 计算参数量
        for p in layer.parameters():
            params += p.numel()

        # 根据层类型计算 FLOPs
        if isinstance(layer, nn.Conv2d):
            flops = self._count_conv2d(layer, input_shape, output_shape)
        elif isinstance(layer, nn.Linear):
            flops = self._count_linear(layer, input_shape, output_shape)
        # 还可以扩展 BatchNorm, ReLU 等层的计算，但通常它们的计算量比起 Conv/Linear 可以忽略不计

        if flops > 0:  # 只记录有计算量的层
            self.layer_stats.append({
                'name': str(layer),
                'type': layer.__class__.__name__,
                'params': params,
                'flops': flops,
                'flops_density': flops / (params + 1e-6)  # 计算密度
            })

    def profile(self):
        # 1. 注册 Hook
        # 我们只遍历有参数的层 (Conv, Linear)
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 注册前向钩子
                self.hooks.append(module.register_forward_hook(self._hook_fn))

        # 2. 运行一次假数据
        self.model.eval()
        dummy_input = torch.zeros(self.input_size).to(self.device)
        self.model.to(self.device)

        with torch.no_grad():
            self.model(dummy_input)

        # 3. 移除 Hook
        for h in self.hooks:
            h.remove()

        return self.layer_stats


# --- 使用示例 ---
if __name__ == "__main__":
    from torchvision.models import vgg16

    # 加载你的模型
    my_model = vgg16()

    # 初始化分析器 (输入尺寸对应 CIFAR-10: 1张图片, 3通道, 32x32)
    profiler = LayerFlopsProfiler(my_model, input_size=(1, 3, 32, 32))

    # 获取每一层的统计数据
    layer_info = profiler.profile()

    print(f"{'Layer Type':<15} | {'Params':<10} | {'FLOPs':<15} | {'Density (FLOPs/Param)':<20}")
    print("-" * 65)

    stats_tensor_list = []

    for info in layer_info:
        print(f"{info['type']:<15} | {info['params']:<10} | {int(info['flops']):<15} | {info['flops_density']:<20.2f}")

        # 收集数据用于 HyperNet 输入
        # [log10(N), log10(FLOPs)] (举例)
        stats_tensor_list.append([info['params'], info['flops']])