import os
import torch
import typing


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    """保存训练断点：
    - model 参数
    - optimizer 状态 (包含动量等)
    - iteration 当前训练的进度
    """

    ckpt_info = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        ckpt_info["optimizer_state_dict"] = optimizer.state_dict()

    if isinstance(out, (str, os.PathLike)):
        out = os.fspath(out)
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)

    torch.save(ckpt_info, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device | None = None,
):
    """
    从指定来源加载模型和优化器状态。
    """
    # 如果未指定设备，且 CUDA 可用，则默认使用 CUDA；否则使用 CPU
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # 加载到指定设备
    checkpoint = torch.load(src, map_location=device)

    # 加载模型权重
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        # 如果是字典但没找到对应键，可能保存的就是 state_dict 本身
        model.load_state_dict(checkpoint)
    else:
        # 处理非字典情况
        model.load_state_dict(checkpoint)

    # 加载优化器状态
    if optimizer is not None:
        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            print("Warning: No optimizer state found in checkpoint.")

    print(f"Checkpoint loaded successfully from {src} to {device}")

    # 为了兼容测试，返回 iteration 或 step
    if isinstance(checkpoint, dict):
        return checkpoint.get("iteration") or checkpoint.get("step")
    return None
