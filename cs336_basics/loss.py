import torch
import torch.nn as nn
import torch.nn.functional as F


def CrossEntropyLoss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    使用 PyTorch 高效实现计算交叉熵损失。
    相比手动实现，F.cross_entropy 进行了算子融合，能极大节省显存并提升速度。
    """
    return F.cross_entropy(logits, targets)


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
