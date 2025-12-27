import torch
import torch.nn as nn
from einops import einsum


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
