import torch
import torch.nn as nn


def CrossEntropyLoss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算交叉熵损失。
    logits: 形状 (B, C)，未归一化的预测分数
    targets: 形状 (B,)，整数标签
    返回: 标量损失值
    """
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    stabilized_logits = logits - max_logits

    lse = torch.log(torch.sum(torch.exp(stabilized_logits), dim=-1))
    predicted_logits = torch.gather(
        stabilized_logits, -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    loss = lse - predicted_logits

    return loss.mean()


def PerplexityLoss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算困惑度损失。
    logits: 形状 (B, C)，未归一化的预测分数
    targets: 形状 (B,)，整数标签
    返回: 标量损失值
    """
    ce_loss = CrossEntropyLoss(logits, targets)
    perplexity = torch.exp(ce_loss)
    return perplexity


def MSELoss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算均方误差损失。
    predictions: 形状 (B, D)，模型预测值
    targets: 形状 (B, D)，真实值
    返回: 标量损失值
    """
    loss = (predictions - targets) ** 2
    return loss.mean()


def L1Loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算L1损失（平均绝对误差）。
    predictions: 形状 (B, D)，模型预测值
    targets: 形状 (B, D)，真实值
    返回: 标量损失值
    """
    loss = torch.abs(predictions - targets)
    return loss.mean()
