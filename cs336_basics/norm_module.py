import torch
import torch.nn as nn
from einops import einsum
from collections.abc import Iterable
import math

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


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], 
                      max_l2_norm: float, 
                      eps: float = 1e-6) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    params_with_grad = [p for p in parameters if p.grad is not None]
    if len(params_with_grad) == 0:
        return

    # 1. 正确计算总范数：平方和的平方根
    total_norm_sq = 0.0
    for p in params_with_grad:
        if p.grad is not None:
            # 使用 detach() 保证不计入计算图
            param_norm = p.grad.detach().norm(2)
            total_norm_sq += param_norm.item() ** 2
    total_norm = math.sqrt(total_norm_sq)

    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        with torch.no_grad():
            for p in params_with_grad:
                if p.grad is not None:
                    p.grad.mul_(clip_coef)


if __name__ == "__main__":
    # 测试场景：模拟一个简单的模型梯度裁剪
    torch.manual_seed(42)
    
    # 1. 创建模拟参数
    p1 = nn.Parameter(torch.randn(2, 3))
    p2 = nn.Parameter(torch.randn(4))
    
    # 手动设置梯度，使其总范数已知
    # p1 梯度全为 1 (6个元素), p2 梯度全为 1 (4个元素)
    # 总范数 = sqrt(6 * 1^2 + 4 * 1^2) = sqrt(10) ≈ 3.162
    p1.grad = torch.ones_like(p1.data)
    p2.grad = torch.ones_like(p2.data)
    
    params = [p1, p2]
    
    def get_current_norm(parameters):
        total_norm_sq = 0.0
        for p in parameters:
            if p.grad is not None:
                total_norm_sq += p.grad.detach().norm(2).item() ** 2
        return math.sqrt(total_norm_sq)

    initial_norm = get_current_norm(params)
    print(f"初始总梯度范数: {initial_norm:.4f} (预期: {math.sqrt(10):.4f})")

    # 2. 测试裁剪情况：设置阈值为 1.0 (小于 3.162)
    max_norm = 1.0
    print(f"\n--- 执行裁剪 (阈值 = {max_norm}) ---")
    gradient_clipping(params, max_l2_norm=max_norm)
    
    clipped_norm = get_current_norm(params)
    print(f"裁剪后总梯度范数: {clipped_norm:.4f} (预期应该接近 {max_norm})")
    print(f"p1 梯度的第一个元素: {p1.grad[0, 0].item():.4f}")
    
    # 3. 测试无需裁剪的情况
    print(f"\n--- 执行裁剪 (阈值 = 10.0, 不应触发) ---")
    # 重新设置梯度为全 1
    p1.grad = torch.ones_like(p1.data)
    p2.grad = torch.ones_like(p2.data)
    
    gradient_clipping(params, max_l2_norm=10.0)
    final_norm = get_current_norm(params)
    print(f"操作后总梯度范数: {final_norm:.4f} (预期保持原始值 {initial_norm:.4f})")
