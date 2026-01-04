import os
import time
from dataclasses import dataclass, field
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# 导入用户自己编写的模型和工具
from cs336_basics.transformer_lm import Transformer_LM
from cs336_basics.optimizer import SimpleAdamW
from cs336_basics.lr_schedule import get_cosine_annealing_lr
from cs336_basics.loss import CrossEntropyLoss
from cs336_basics.dataloader import simple_dataloader
from cs336_basics.norm_module import gradient_clipping
from cs336_basics.utils import save_checkpoint, load_checkpoint


@dataclass
class TrainingConfig:
    # 模型超参数
    vocab_size: int = 50257  # 根据实际 tokenizer 调整
    context_length: int = 128
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 1024
    rope_theta: float = 10000.0

    # 训练超参数
    batch_size: int = 32
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    max_iters: int = 5000
    warmup_iters: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # IO 与 日志参数
    train_data_path: str = "data/TinyStoriesV2-GPT4-valid.bin"
    valid_data_path: str = "data/TinyStoriesV2-GPT4-valid.bin"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    eval_interval: int = 500
    log_interval: int = 10
    eval_iters: int = 50

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_data_loader(file_path, config, random_start=True):
    """
    使用 np.memmap 高效加载大型训练和验证数据集至内存
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data file not found: {file_path}. Please run preprocess_data.py first."
        )

    # 使用 uint16 读取二进制文件 (与 preprocess_data.py 保持一致)
    data = np.memmap(file_path, dtype=np.uint16, mode="r")
    return simple_dataloader(
        dataset=data,
        batch_size=config.batch_size,
        context_length=config.context_length,
        device=config.device,
        random_start=random_start,
    )


@torch.no_grad()
def estimate_loss(model, train_loader, valid_loader, config):
    """
    定期计算训练集和验证集的平均损失
    """
    out = {}
    model.eval()
    for split, loader in [("train", train_loader), ("valid", valid_loader)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = next(loader)
            logits = model(x)
            # 展平 logits 和 targets 以匹配 CrossEntropyLoss 的输入要求
            B, L, V = logits.shape
            loss = CrossEntropyLoss(logits.view(B * L, V), y.view(B * L))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(config: TrainingConfig):
    # 准备目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # TensorBoard 记录器
    writer = SummaryWriter(log_dir=config.log_dir)

    device = torch.device(config.device)
    print(f"Using device: {device}")

    # 初始化模型
    model = Transformer_LM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=device,
    )
    model.to(device)

    # 初始化优化器
    optimizer = SimpleAdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # 初始化数据加载器
    try:
        train_loader = get_data_loader(
            config.train_data_path, config, random_start=True
        )
        valid_loader = get_data_loader(
            config.valid_data_path, config, random_start=True
        )
    except FileNotFoundError as e:
        print(e)
        return

    iter_num = 0
    best_val_loss = float("inf")

    print("Starting training loop...")
    t0 = time.time()

    while iter_num <= config.max_iters:
        # 1. 计算当前学习率 (余弦退火)
        lr = get_cosine_annealing_lr(
            iter_num,
            config.learning_rate,
            config.min_learning_rate,
            config.warmup_iters,
            config.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 2. 定期评估与保存检查点
        if iter_num > 0 and iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, train_loader, valid_loader, config)
            print(
                f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['valid']:.4f}, lr {lr:.2e}"
            )

            # 记录到 TensorBoard
            writer.add_scalar("Loss/train", losses["train"], iter_num)
            writer.add_scalar("Loss/valid", losses["valid"], iter_num)
            writer.add_scalar("LearningRate", lr, iter_num)

            # 如果是最佳模型则单独保存
            if losses["valid"] < best_val_loss:
                best_val_loss = losses["valid"]
                save_path = os.path.join(config.checkpoint_dir, "best_model.pt")
                save_checkpoint(model, optimizer, iter_num, save_path)
                print(f"Saved new best model to {save_path}")

            # 定期保存普通检查点
            checkpoint_path = os.path.join(config.checkpoint_dir, f"ckpt_latest.pt")
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)

        # 3. 训练步骤
        x, y = next(train_loader)
        logits = model(x)

        # 展平以便计算损失
        B, L, V = logits.shape
        loss = CrossEntropyLoss(logits.view(B * L, V), y.view(B * L))

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if config.grad_clip > 0:
            gradient_clipping(model.parameters(), config.grad_clip)

        optimizer.step()

        # 4. 控制台日志记录
        if iter_num % config.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            lossf = loss.item()
            print(f"Iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
            writer.add_scalar("Loss/iter", lossf, iter_num)

        iter_num += 1

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    # 用户可以在这里修改配置
    config = TrainingConfig()
    train(config)
