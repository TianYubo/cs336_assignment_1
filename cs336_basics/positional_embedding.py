import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化 RoPE 模块，预计算 Cos 和 Sin 表。
        """
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        # 1. 计算频率向量 (Frequencies)
        # 根据公式 theta_i = 1 / (base ^ (2i / d)), 生成 [theta_0, theta_1, ... theta_{d/2-1}]
        # 这里的 2i 对应代码中的 arange(0, d_k, 2)
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        # 2. 生成位置索引 [0, 1, 2, ..., max_seq_len-1]
        position_idx = torch.arange(max_seq_len, device=device).float()

        # 3. 计算角度表 (Angles)
        # 外积操作：位置 m * 频率 theta_i
        # 结果形状: (max_seq_len, d_k / 2)
        angles = torch.outer(position_idx, freqs)

        # 4. 扩展角度表以匹配输入维度
        # RoPE 是将向量两两配对旋转，所以我们需要把角度重复一次
        # 变成了 [theta_0, theta_1, ..., theta_0, theta_1, ...] 的形式
        # 结果形状: (max_seq_len, d_k)
        angles = torch.cat((angles, angles), dim=-1)

        # 5. 预计算并缓存 Cos 和 Sin (注册为 buffer，不算作模型参数)
        self.register_buffer("cos_cached", angles.cos())
        self.register_buffer("sin_cached", angles.sin())

    def forward(self, x, position_ids):
        """
        x: 输入张量，形状 (batch_size, seq_len, d_k)
        position_ids: 位置索引，形状 (batch_size, seq_len)
        """
        # 1. 根据输入的位置索引，从缓存中“切片”取出对应的 Cos 和 Sin
        # 形状变为: (batch_size, seq_len, d_k)
        # 这里使用了 PyTorch 的高级索引功能
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]

        # 2. 应用旋转
        # 这里的核心公式是: x' = x * cos + rotate_half(x) * sin
        return self.apply_rotary_emb(x, cos, sin)

    def apply_rotary_emb(self, x, cos, sin):
        """
        最原始的旋转逻辑实现。
        为了模拟二维旋转 (x, y) -> (x cos - y sin, x sin + y cos)，
        我们需要构造一个“旋转后的 x”，即 (-y, x)。
        """
        # 把 x 拆分成前半部分 (x1) 和后半部分 (x2)
        x1, x2 = x.chunk(2, dim=-1)

        # 构造 (-x2, x1)，这相当于把向量旋转了 90 度
        # 拼接回去，形状还是 (batch_size, seq_len, d_k)
        x_rotated = torch.cat((-x2, x1), dim=-1)

        # 标准 RoPE 公式：
        # 原向量 * cos + 旋转90度向量 * sin
        return (x * cos) + (x_rotated * sin)
