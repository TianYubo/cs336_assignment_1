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
        x: 输入张量，形状 (..., num_heads, seq_len, d_k) 或 (..., seq_len, d_k)
        position_ids: 位置索引，形状 (..., seq_len)
        """
        # 1. 根据输入的位置索引，从缓存中“切片”取出对应的 Cos 和 Sin
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]

        # 如果 x 有 num_heads 维度，而 cos 没有，则需要 unsqueeze
        if x.ndim > cos.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        # 2. 应用旋转
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


class RotaryPositionalEmbeddingAdjacent(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        相邻元素两两配对的 RoPE 实现 (x0, x1), (x2, x3) ...
        这是 RoPE 原始论文中描述的排列方式。
        """
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        # 1. 计算频率向量
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        position_idx = torch.arange(max_seq_len, device=device)
        angles = torch.outer(position_idx, freqs)

        # 2. 扩展角度表：相邻重复 [theta_0, theta_0, theta_1, theta_1, ...]
        # 形状: (max_seq_len, d_k)
        angles = angles.repeat_interleave(2, dim=-1)

        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x, position_ids):
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]

        # 如果 x 有 num_heads 维度，而 cos 没有，则需要 unsqueeze
        if x.ndim > cos.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        return self.apply_rotary_emb(x, cos, sin)

    def apply_rotary_emb(self, x, cos, sin):
        # 构造旋转向量: [-x1, x0, -x3, x2, ...]
        # 这种实现需要通过 reshape 来交换相邻元素
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        x0 = x_reshaped[..., 0]
        x1 = x_reshaped[..., 1]

        # 拼接成 [-x1, x0] 的形式并还原形状
        x_rotated = torch.stack((-x1, x0), dim=-1).view_as(x)

        return (x * cos) + (x_rotated * sin)


def _test_shapes_and_cache():
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 4
    d_k = 8
    theta = 10000.0
    max_seq_len = 16

    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    x = torch.randn(batch_size, seq_len, d_k)
    position_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    y = rope(x, position_ids)

    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    assert rope.cos_cached.shape == (max_seq_len, d_k)
    assert rope.sin_cached.shape == (max_seq_len, d_k)
    print("test_shapes_and_cache: OK")


def _test_identity_at_pos0():
    torch.manual_seed(0)
    batch_size = 1
    seq_len = 1
    d_k = 8
    theta = 10000.0
    max_seq_len = 8

    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    x = torch.randn(batch_size, seq_len, d_k)
    position_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    y = rope(x, position_ids)

    # position 0 => angle 0 => cos=1, sin=0 so output should equal input
    if not torch.allclose(x, y, atol=1e-6):
        max_diff = (x - y).abs().max().item()
        raise AssertionError(f"identity check failed, max diff: {max_diff}")
    print("test_identity_at_pos0: OK")


def _test_norm_preservation():
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 5
    d_k = 8
    theta = 10000.0
    max_seq_len = 16

    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    x = torch.randn(batch_size, seq_len, d_k)
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    y = rope(x, position_ids)

    # RoPE is a rotation, so per-token vector norms should be preserved
    x_norm = torch.linalg.norm(x, dim=-1)
    y_norm = torch.linalg.norm(y, dim=-1)
    if not torch.allclose(x_norm, y_norm, atol=1e-6):
        max_diff = (x_norm - y_norm).abs().max().item()
        raise AssertionError(f"norm check failed, max diff: {max_diff}")
    print("test_norm_preservation: OK")


def _test_compare_implementations():
    torch.manual_seed(0)
    batch_size = 1
    seq_len = 2
    d_k = 4
    theta = 10000.0
    max_seq_len = 8

    x = torch.randn(batch_size, seq_len, d_k)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    rope_split = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    rope_adj = RotaryPositionalEmbeddingAdjacent(theta, d_k, max_seq_len)

    y_split = rope_split(x, position_ids)
    y_adj = rope_adj(x, position_ids)

    print("Input x:\n", x)
    print("Output (Split):\n", y_split)
    print("Output (Adjacent):\n", y_adj)

    # Both should preserve norm
    assert torch.allclose(
        torch.linalg.norm(x, dim=-1), torch.linalg.norm(y_split, dim=-1)
    )
    assert torch.allclose(
        torch.linalg.norm(x, dim=-1), torch.linalg.norm(y_adj, dim=-1)
    )
    print(
        "test_compare_implementations: OK (Both preserve norms, but outputs differ in ordering)"
    )


if __name__ == "__main__":
    _test_shapes_and_cache()
    _test_identity_at_pos0()
    _test_norm_preservation()
    _test_compare_implementations()
