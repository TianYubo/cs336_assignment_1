import math
import matplotlib.pyplot as plt

def get_cosine_annealing_lr(
    t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int
) -> float:
    """
    计算带预热（Warmup）的余弦退火（Cosine Annealing）学习率。

    这是一个非常经典的学习率调度策略，常用于训练大型语言模型（如 GPT 系列）。

    参数:
        t: 当前迭代步数 (current step)
        alpha_max: 学习率的最大值（预热结束后的值）
        alpha_min: 学习率的最小值（最终退火到的值）
        T_w: 预热阶段的步数 (warmup steps)
        T_c: 整个调度过程的总步数 (total steps)

    返回:
        当前步数 t 对应的学习率
    """

    # 1. 线性预热阶段 (Linear Warmup)
    # 如果当前步数小于预热步数，学习率从 0 线性增加到 alpha_max
    if t < T_w:
        return (t / T_w) * alpha_max

    # 2. 超过总步数 (Beyond Total Steps)
    # 如果当前步数超过了设定的总步数，通常保持最小学习率 alpha_min
    if t >= T_c:
        return alpha_min

    # 3. 余弦退火阶段 (Cosine Annealing)
    # 计算当前处于余弦退火阶段的比例 (从 0 到 1)
    # t - T_w 是进入退火阶段后的步数
    # T_c - T_w 是整个退火阶段的总长度
    progress = (t - T_w) / (T_c - T_w)

    # 余弦退火公式:
    # alpha_t = alpha_min + 0.5 * (1 + cos(pi * progress)) * (alpha_max - alpha_min)
    #
    # 原理拆解:
    # - math.pi * progress: 弧度从 0 变化到 pi
    # - math.cos(...): 值从 1 变化到 -1
    # - 1 + math.cos(...): 值从 2 变化到 0
    # - 0.5 * (1 + math.cos(...)): 值从 1 变化到 0 (这就是我们的衰减系数)

    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    lr = alpha_min + cosine_decay * (alpha_max - alpha_min)

    return lr


# 范例用法
if __name__ == "__main__":

    # 设定比较的参数
    configs = [
        {
            "name": "Default (a_max=6e-4, a_min=6e-5, T_w=100, T_c=1000)",
            "a_max": 6e-4,
            "a_min": 6e-5,
            "warmup_steps": 100,
            "total_steps": 1000,
        },
        {
            "name": "Longer Warmup (T_w=300)",
            "a_max": 6e-4,
            "a_min": 6e-5,
            "warmup_steps": 300,
            "total_steps": 1000,
        },
        {
            "name": "Lower alpha_min (a_min=0)",
            "a_max": 6e-4,
            "a_min": 0,
            "warmup_steps": 100,
            "total_steps": 1000,
        },
    ]

    plt.figure(figsize=(10, 6))

    for config in configs:
        steps = list(range(config["total_steps"] + 101))
        lrs = [
            get_cosine_annealing_lr(
                t,
                config["a_max"],
                config["a_min"],
                config["warmup_steps"],
                config["total_steps"],
            )
            for t in steps
        ]
        plt.plot(steps, lrs, label=config["name"])

    plt.title("Cosine Annealing with Warmup - Parameter Comparison")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()

    # 打印一些关键点的数值作为参考
    print("Example (Default Config):")
    c = configs[0]
    for t in [0, 50, 100, 550, 1000]:
        lr = get_cosine_annealing_lr(
            t, c["a_max"], c["a_min"], c["warmup_steps"], c["total_steps"]
        )
        print(f"  Step {t:4d}: {lr:.2e}")
