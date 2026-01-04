from collections.abc import Callable, Iterable
from typing import Optional
import torch
from torch.optim import Optimizer
import math


class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get(
                    "t", 0
                )  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.

                # Update weight tensor in-place.
                p.data -= lr / math.sqrt(t + 1) * grad

                state["t"] = t + 1  # Increment iteration number.

        return loss


class SimpleAdamW(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    ):
        """
        Custom AdamW Optimizer.

        Args:
            params: 待优化的参数迭代器。
            lr: 学习率 (eta)。
            betas: 用于计算一阶和二阶矩的系数系数。
            eps: 分母数值稳定性项。
            weight_decay: 权重衰减系数 (lambda)。
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(SimpleAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行一步参数更新。
        """

        # 闭包，用于重新计算损失（如果需要）
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # 1. 获取当前梯度
                grad = p.grad

                # 2. 状态初始化（存储动量等变量）
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # 一阶矩
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # 二阶矩
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # --- 核心步骤开始 ---

                # 3. 权重衰减解耦 (Weight Decay Decoupling)
                # 与 Adam 不同，AdamW 直接在更新前对参数进行衰减，不将衰减项加入梯度计算
                # 公式: theta = theta - lr * wd * theta
                p.mul_(1 - lr * wd)

                # 4. 更新一阶矩 (Momentum)
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 5. 更新二阶矩 (Adaptive scale)
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 6. 计算偏差修正 (Bias Correction)
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t

                # 7. 更新参数
                # 计算自适应步长：m_hat / (sqrt(v_hat) + eps)
                # 这里的 step_size 将 lr 和修正系数结合在一起
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)

                # theta = theta - step_size * (exp_avg / denom)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class NaiveAdamW(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    ):
        """
        A deliberately straightforward AdamW implementation for learning.

        Args:
            params: 待优化的参数迭代器。
            lr: 学习率 (eta)。
            betas: 一阶/二阶矩系数 (beta1, beta2)。
            eps: 分母数值稳定性项。
            weight_decay: 权重衰减系数 (lambda)。
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行一步参数更新（更直观的写法，重在易读）。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["v"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                t = state["step"]
                m = state["m"]
                v = state["v"]

                # 1) 解耦权重衰减：直接缩小参数本身，避免把 L2 项混进梯度
                p.mul_(1 - lr * wd)

                # 2) 一阶矩：对梯度做指数滑动平均，保留“方向”的惯性
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 3) 二阶矩：对梯度平方做指数滑动平均，度量“尺度/方差”
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 4) 偏差修正：早期 m、v 偏小，用 (1 - beta^t) 纠正
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                # 5) 参数更新：按元素缩放学习率，梯度大则步子小
                update = m_hat / (v_hat.sqrt() + eps)
                p.add_(update, alpha=-lr)

        return loss


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e-2)

    for t in range(100):
        opt.zero_grad()
        loss = (weights**2).mean()

        print(loss.cpu().item())
        loss.backward()
        opt.step()
