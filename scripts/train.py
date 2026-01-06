import os
import time
import random
from dataclasses import dataclass, field, asdict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

# 导入用户自己编写的模型和工具
from cs336_basics.transformer_lm import Transformer_LM, PostNormTransformer_LM
from cs336_basics.optimizer import SimpleAdamW
from cs336_basics.lr_schedule import get_cosine_annealing_lr
from cs336_basics.loss import CrossEntropyLoss
from cs336_basics.dataloader import simple_dataloader
from cs336_basics.norm_module import gradient_clipping
from cs336_basics.utils import save_checkpoint, load_checkpoint


@dataclass
class TrainingConfig:
    # 模型超参数
    vocab_path: str = "tests/fixtures/gpt2_vocab.json"
    vocab_size: int = None  # 将在运行时根据 vocab_path 自动确定
    context_length: int = 512  # 推荐: 128-512 (对于 TinyStories, 256-512 效果更好)
    d_model: int = 768  # 推荐: 128-768 (需被 num_heads 整除)
    num_layers: int = 8  # 推荐: 4-12 (层数多利于理解复杂语法)
    num_heads: int = (
        16  # 推荐: 4-12 (每个 head 的维度 d_model/num_heads 建议在 32-128 之间)
    )
    d_ff: int = 1344  # 推荐: 4 * d_model (标准 Transformer 比例)
    rope_theta: float = 10000.0  # 常用值: 10000.0 (外推需求大时可调大)
    # rope_theta = None

    # 训练超参数
    batch_size: int = (
        32  # 推荐: 16-128 (视显存而定，总 batch tokens = batch_size * context_length)
    )
    learning_rate: float = (
        5e-4  # 推荐: 3e-4 到 1e-3 (小模型 LR 可稍大，大模型通常用 3e-4 或更小)
    )
    min_learning_rate: float = 6e-5  # 推荐: 0.1 * learning_rate 或更低
    max_iters: int = (
        20000  # 推荐: 视 Loss 曲线而定，TinyStories 充分训练通常需 20k+ steps
    )
    warmup_iters: int = 500  # 推荐: 5%-10% of max_iters
    weight_decay: float = 0.1  # 推荐: 0.01 - 0.1 (用于防止过拟合)
    grad_clip: float = 1.0  # 推荐: 1.0 (防止梯度爆炸，必设)
    use_amp: bool = True  # 是否使用混合精度训练 (GTX 1660 Ti 强烈建议开启)
    amp_dtype: torch.dtype = torch.bfloat16

    # IO 与 日志参数
    train_data_path: str = "/root/autodl-tmp/data/TinyStoriesV2-GPT4-train.bin"
    valid_data_path: str = "/root/autodl-tmp/data/TinyStoriesV2-GPT4-valid.bin"
    checkpoint_dir: str = "checkpoints/transformer-run-512-4-4-1344"
    log_dir: str = "logs"
    eval_interval: int = 1000  # 建议: 每 500-1000 步评估一次
    log_interval: int = 50  # 建议: 10-50 步记录一次日志
    eval_iters: int = 32      # 评估时使用的 batch 数量
    compile: bool = True  # 是否使用 torch.compile 加速 (首次运行有编译延迟)
    gpu_memory_debug: bool = False  # 是否开启显存占用深度分析

    # WandB 参数
    use_wandb: bool = True
    wandb_project: str = "cs336-assignment-1"
    wandb_run_name: str = field(
        default_factory=lambda: f"Bigger Baseline"
    )

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    """
    固定所有随机种子以确保实验的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保 CUDA 卷积算子是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置环境变量以确保某些特定的 CUDA 算子是确定性的
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


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
            # 在评估时也使用 autocast 以保持内存效率
            with torch.amp.autocast("cuda", enabled=config.use_amp, dtype=config.amp_dtype):
                logits = model(x)
                B, L, V = logits.shape
                loss = CrossEntropyLoss(logits.view(B * L, V), y.view(B * L))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(config: TrainingConfig):
    # 固定随机种子
    set_seed(config.seed)

    # 准备目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # 初始化 WandB
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=asdict(config),
        )

    # TensorBoard 记录器
    writer = SummaryWriter(log_dir=config.log_dir)

    device = torch.device(config.device)
    print(f"Using device: {device}")

    # 如果开启了显存调试，记录显存历史
    if config.gpu_memory_debug:
        torch.cuda.memory._record_memory_history(max_entries=100000)

    # 自动获取并校验词表大小
    if config.vocab_size is None:
        import json

        if os.path.exists(config.vocab_path):
            with open(config.vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
                config.vocab_size = len(vocab)
            print(
                f"Successfully loaded vocab from {config.vocab_path}. Vocab size: {config.vocab_size}"
            )
        else:
            # 如果没找到词表文件，则报错提醒，避免模型参数设置错误
            raise FileNotFoundError(
                f"Vocab file not found at {config.vocab_path}. "
                "Please provide a valid vocab_path or set vocab_size manually."
            )

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
    # model = PostNormTransformer_LM(
    #     vocab_size=config.vocab_size,
    #     context_length=config.context_length,
    #     d_model=config.d_model,
    #     num_layers=config.num_layers,
    #     num_heads=config.num_heads,
    #     d_ff=config.d_ff,
    #     rope_theta=config.rope_theta,
    #     device=device,
    # )
    model.to(device)

    # 使用 torch.compile 加速模型 (PyTorch 2.0+)
    if config.compile:
        print("Compiling model (this may take a minute)...")
        # 针对 4090，建议使用 default 或 max-autotune。
        # 注意：reduce-overhead 模式会预分配大量显存。
        model = torch.compile(model)
    
    torch.set_float32_matmul_precision('high')
    
    # 初始化优化器
    optimizer = SimpleAdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # 初始化混合精度训练的 Scaler
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

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
    total_time = 0.0  # 累计挂钟时间

    print("Starting training loop...")
    start_time = time.time()
    t0 = start_time

    # 如果开启了显存调试，初始化 Profiler
    prof = None
    if config.gpu_memory_debug:
        print("GPU Memory Debug enabled. Profiling iters 10-14...")
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=10, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()

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

            # 记录到 WandB
            if config.use_wandb:
                wandb.log(
                    {
                        "val/loss": losses["valid"],
                        "train/loss_eval": losses["train"],
                        "lr": lr,
                        "wall_clock_time": time.time() - start_time,
                    },
                    step=iter_num,
                )

            # 如果是最佳模型则单独保存
            if losses["valid"] < best_val_loss:
                best_val_loss = losses["valid"]
                save_path = os.path.join(config.checkpoint_dir, "best_model.pt")
                # 如果模型被编译了，保存原始模型以保持 state_dict 干净
                raw_model = model._orig_mod if config.compile else model
                save_checkpoint(raw_model, optimizer, iter_num, save_path)
                print(f"Saved new best model to {save_path}")

            # 定期保存普通检查点
            checkpoint_path = os.path.join(config.checkpoint_dir, f"ckpt_latest.pt")
            raw_model = model._orig_mod if config.compile else model
            save_checkpoint(raw_model, optimizer, iter_num, checkpoint_path)

        # 3. 训练步骤
        x, y = next(train_loader)

        # 使用 autocast 进行混合精度训练
        with torch.amp.autocast("cuda", enabled=config.use_amp, dtype=config.amp_dtype):
            logits = model(x)
            # 展平以便计算损失
            B, L, V = logits.shape
            loss = CrossEntropyLoss(logits.view(B * L, V), y.view(B * L))

        optimizer.zero_grad()
        # 使用 scaler 缩放损失并反向传播
        scaler.scale(loss).backward()

        # 梯度裁剪 (在 unscale 之后进行)
        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            gradient_clipping(model.parameters(), config.grad_clip)

        # 优化器步进
        scaler.step(optimizer)
        scaler.update()

        # Profiler 步进
        if prof:
            prof.step()
            # 在 profiling 结束后的那一步保存显存快照 (pickle 格式)
            if iter_num == 14:
                print("Saving memory snapshot...")
                snapshot = torch.cuda.memory._snapshot()
                import pickle
                with open(os.path.join(config.log_dir, "mem_snapshot.pickle"), "wb") as f:
                    pickle.dump(snapshot, f)
                prof.stop()
                prof = None

        # 4. 控制台日志记录
        if iter_num % config.log_interval == 0:
            t1 = time.time()
            lossf = loss.item()
            wall_time = t1 - start_time
            allocated = torch.cuda.memory_allocated() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2

            if iter_num > 0:
                dt = t1 - t0
                # 计算吞吐量 (Tokens per second)
                tokens_per_iter = x.numel()
                tokens_per_sec = (config.log_interval * tokens_per_iter) / dt

                # 计算预计剩余时间 (ETA)
                remaining_iters = config.max_iters - iter_num
                eta_seconds = (dt / config.log_interval) * remaining_iters
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

                print(
                    f"Iter {iter_num:5d}/{config.max_iters} | "
                    f"Loss: {lossf:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Time: {dt*1000/config.log_interval:.2f}ms/it | "
                    f"Tok/s: {tokens_per_sec:7.1f} | "
                    f"ETA: {eta_str} | "
                    f"Mem: {allocated:.0f}MB/{peak:.0f}MB"
                )
            else:
                print(f"Iter {iter_num:5d}/{config.max_iters} | Loss: {lossf:.4f} | LR: {lr:.2e}")

            t0 = t1
            writer.add_scalar("Loss/iter", lossf, iter_num)
            writer.add_scalar("Time/wall_clock", wall_time, iter_num)
            writer.add_scalar("Memory/Allocated", allocated, iter_num)
            writer.add_scalar("Memory/Peak", peak, iter_num)

            if config.use_wandb:
                metrics = {
                    "train/loss": lossf,
                    "train/lr": lr,
                    "train/mem_allocated": allocated,
                    "train/mem_peak": peak,
                    "wall_clock_time": wall_time,
                }
                if iter_num > 0:
                    metrics.update({
                        "train/iter_dt": dt / config.log_interval,
                        "train/tokens_per_sec": tokens_per_sec,
                    })
                wandb.log(metrics, step=iter_num)

        iter_num += 1

    if prof:
        prof.stop()
    writer.close()
    if config.use_wandb:
        wandb.finish()
    print("Training finished.")


if __name__ == "__main__":
    # 用户可以在这里修改配置
    config = TrainingConfig()
    train(config)
