import torch
import torch.nn as nn
from einops import einsum
from .softmax import softmax_function
from .positional_embedding import RotaryPositionalEmbeddingAdjacent
from torch import Tensor
from jaxtyping import Float, Int


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    计算缩放点积注意力 (Scaled Dot-Product Attention)。

    参数:
        query (torch.Tensor): 查询张量，形状为 (..., seq_len_q, d_k)。
        key (torch.Tensor): 键张量，形状为 (..., seq_len_k, d_k)。
        value (torch.Tensor): 值张量，形状为 (..., seq_len_v, d_v)，通常 seq_len_k == seq_len_v。
        mask (torch.Tensor, optional): 可选的掩码张量，形状为 (..., seq_len_q, seq_len_k)。
    返回:
        torch.Tensor: 注意力输出张量，形状为 (..., seq_len_q, d_v)。
    """
    d_k = query.shape[-1]

    # 计算注意力分数 (Attention Scores)
    # 注意：这里使用 query.dtype，因为 scores 还没定义
    scale = torch.rsqrt(torch.tensor(d_k, dtype=query.dtype, device=query.device))

    scores = (
        einsum(
            query,
            key,
            "... i d, ... j d -> ... i j",
        )
        * scale
    )

    # 应用掩码 (Masking)
    if mask is not None:
        # 将 mask 为 0 的地方填充为负无穷，这样 softmax 之后权重几乎就是 0
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Softmax 归一化得到注意力权重
    attn_weights = softmax_function(scores, dim=-1)

    # 加权求和得到最终输出
    output = einsum(
        attn_weights,
        value,
        "... i j, ... j v -> ... i v",
    )

    return output


def causal_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, "d_k d_in"],
    k_proj_weight: Float[Tensor, "d_k d_in"],
    v_proj_weight: Float[Tensor, "d_v d_in"],
    o_proj_weight: Float[Tensor, "d_model d_v"],
    in_features: Float[Tensor, "... sequence_length d_in"],
):
    """
    多头因果自注意力机制 (Causal Multi-Head Self-Attention)。

    参数:
        d_model (int): 模型的隐藏维度。
        num_heads (int): 注意力头的数量。
        q_proj_weight (Float[Tensor, "d_k d_in"]): 查询的线性投影权重。
        k_proj_weight (Float[Tensor, "d_k d_in"]): 键的线性投影权重。
        v_proj_weight (Float[Tensor, "d_v d_in"]): 值的线性投影权重。
        o_proj_weight (Float[Tensor, "d_model d_v"]): 输出的线性投影权重。
        in_features (Float[Tensor, "... sequence_length d_in"]): 输入特征张量。

    返回:
        Float[Tensor, "... sequence_length d_model"]: 多头自注意力的输出张量。
    """
    batch_dims = in_features.shape[:-2]
    seq_len = in_features.shape[-2]
    d_in = in_features.shape[-1]

    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]

    Q = einsum(
        q_proj_weight, in_features, "d_k d_in, ... seq_len d_in -> ... seq_len d_k"
    )
    K = einsum(
        k_proj_weight, in_features, "d_k d_in, ... seq_len d_in -> ... seq_len d_k"
    )
    V = einsum(
        v_proj_weight, in_features, "d_v d_in, ... seq_len d_in -> ... seq_len d_v"
    )

    Q = Q.view(*batch_dims, seq_len, num_heads, d_k // num_heads).transpose(-3, -2)
    K = K.view(*batch_dims, seq_len, num_heads, d_k // num_heads).transpose(-3, -2)
    V = V.view(*batch_dims, seq_len, num_heads, d_v // num_heads).transpose(-3, -2)

    # 创建因果掩码 (Causal Mask)
    mask = torch.tril(
        torch.ones((seq_len, seq_len), device=in_features.device)
    ).unsqueeze(0)

    # 缩放点积注意力
    attn_output = scaled_dot_product_attention(
        Q, K, V, mask=mask
    )  # (..., num_heads, seq_len, d_v // num_heads)
    attn_output = (
        attn_output.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, d_v)
    )  # (..., num_heads, seq_len, d_v // num_heads) -> (..., seq_len, d_v)

    output = einsum(
        o_proj_weight,
        attn_output,
        "d_model d_v, ... seq_len d_v -> ... seq_len d_model",
    )

    return output


def causal_multihead_self_attention_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, "d_k d_in"],
    k_proj_weight: Float[Tensor, "d_k d_in"],
    v_proj_weight: Float[Tensor, "d_v d_in"],
    o_proj_weight: Float[Tensor, "d_model d_v"],
    in_features: Float[Tensor, "... sequence_length d_in"],
    token_positions: Int[Tensor, "... sequence_length"] | None = None,
):
    """
    多头因果自注意力机制 (Causal Multi-Head Self-Attention) 使用 RoPE 位置编码。

    参数:
        d_model (int): 模型的隐藏维度。
        num_heads (int): 注意力头的数量。
        max_seq_len (int): 最大序列长度。
        theta (float): RoPE 的频率基数。
        q_proj_weight (Float[Tensor, "d_k d_in"]): 查询的线性投影权重。
        k_proj_weight (Float[Tensor, "d_k d_in"]): 键的线性投影权重。
        v_proj_weight (Float[Tensor, "d_v d_in"]): 值的线性投影权重。
        o_proj_weight (Float[Tensor, "d_model d_v"]): 输出的线性投影权重。
        in_features (Float[Tensor, "... sequence_length d_in"]): 输入特征张量。
        token_positions (Int[Tensor, "... sequence_length"], optional): 位置索引张量。

    返回:
        Float[Tensor, "... sequence_length d_model"]: 多头自注意力的输出张量。
    """

    rope = RotaryPositionalEmbeddingAdjacent(
        theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len
    )

    batch_dims = in_features.shape[:-2]
    seq_len = in_features.shape[-2]
    d_in = in_features.shape[-1]

    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]

    Q = einsum(
        q_proj_weight, in_features, "d_k d_in, ... seq_len d_in -> ... seq_len d_k"
    )
    K = einsum(
        k_proj_weight, in_features, "d_k d_in, ... seq_len d_in -> ... seq_len d_k"
    )
    V = einsum(
        v_proj_weight, in_features, "d_v d_in, ... seq_len d_in -> ... seq_len d_v"
    )

    Q = Q.view(*batch_dims, seq_len, num_heads, d_k // num_heads).transpose(-3, -2)
    K = K.view(*batch_dims, seq_len, num_heads, d_k // num_heads).transpose(-3, -2)
    V = V.view(*batch_dims, seq_len, num_heads, d_v // num_heads).transpose(-3, -2)

    if token_positions is not None:
        position_ids = token_positions
        Q = rope(Q, position_ids)
        K = rope(K, position_ids)

    # 创建因果掩码 (Causal Mask)
    mask = torch.tril(
        torch.ones((seq_len, seq_len), device=in_features.device)
    ).unsqueeze(0)

    # 缩放点积注意力
    attn_output = scaled_dot_product_attention(
        Q, K, V, mask=mask
    )  # (..., num_heads, seq_len, d_v // num_heads)
    attn_output = (
        attn_output.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, d_v)
    )  # (..., num_heads, seq_len, d_v // num_heads) -> (..., seq_len, d_v)

    output = einsum(
        o_proj_weight,
        attn_output,
        "d_model d_v, ... seq_len d_v -> ... seq_len d_model",
    )

    return output
