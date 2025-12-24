import torch
import torch.nn as nn


def softmax_function(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    计算输入张量 x 在指定维度 dim 上的 Softmax。
    Softmax 函数将输入张量的值映射到 (0, 1) 区间，并且所有值的和为 1。

    参数:
        x (torch.Tensor): 输入张量。
        dim (int): 指定进行 Softmax 计算的维度，默认为最后一个维度。
    返回:
        torch.Tensor: 经过 Softmax 计算后的张量，形状与输入张量相同。
    """
    # 1. 减去最大值以提高数值稳定性
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_stable = x - x_max

    # 2. 计算指数
    exp_x = torch.exp(x_stable)

    # 3. 计算归一化因子（每个切片的和）
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)

    # 4. 计算 Softmax
    softmax_x = exp_x / sum_exp_x

    return softmax_x
