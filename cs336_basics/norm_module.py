import torch
import torch.nn as nn
from einops import einsum

"""
RMSNorm implementation.
FLOPs 分析：
- 针对一个长度为 D 的向量，RMSNorm 主要涉及以下计算步骤：
    1. 计算平方：D 次乘法
    2. 计算均值：D-1 次加法 + 1 次除法
    3. 加偏移量 + self.eps: 1 次加法
    4. 计算平方根的倒数（rsqrt）：1 次平方根 + 1 次求导
    5. 归一化乘以权重：D 次乘法
    6. 仿射变换（乘以权重）：D 次乘法
- 总计：
    - 乘法：3D + 1
    - 加法：D
    - 除法：1
    - 平方根：1
- 因此，RMSNorm 的总 FLOPs 约为 4D + 4

对于（B, L, D) 的输入，RMSNorm 的总 FLOPs 约为 B * L * (4D + 4)，看作为 4BLD FLOPs
"""


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        gain_tensor = torch.ones(d_model, dtype=dtype)
        if device is not None:
            gain_tensor = gain_tensor.to(device)
        self.weight = nn.Parameter(gain_tensor)

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # # 方法1：最直接的方式
        # rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # rms = 1.0 / rms

        # 方法2：使用 rsqrt (reciprocal square root) 更快
        rms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        rms_norm = x * rms * self.weight  # 注意这里是乘法

        # rms_norm = x / rms * self.gain
        return rms_norm.to(in_dtype)
