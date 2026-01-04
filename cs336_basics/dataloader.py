import numpy as np
import torch
from typing import Generator, Union
import numpy.typing as npt


def simple_dataloader(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: Union[str, torch.device] = "cpu",
    random_start: bool = True,
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    支持无限随机采样或无限顺序采样的数据加载器。
    """
    n = len(dataset)
    max_start_idx = n - context_length - 1
    # 维护一个持久化指针，用于记录顺序采样的位置
    start_ptr = 0

    while True:
        if random_start:
            # 模式 1：随机采样起始位置
            ix = np.random.randint(0, max_start_idx, size=batch_size)
        else:
            # 模式 2：顺序循环采样
            # 如果剩余数据不足以构造一个完整的 batch，则重置指针回到开头
            if start_ptr + batch_size > max_start_idx:
                start_ptr = 0

            # 构造当前 batch 的起始索引：从 start_ptr 开始取 batch_size 个连续索引
            ix = np.arange(start_ptr, start_ptr + batch_size)
            # 更新指针位置
            start_ptr += batch_size

        # 根据生成的起始索引 ix 构造 batch
        x_batch = np.stack([dataset[i : i + context_length] for i in ix])
        y_batch = np.stack([dataset[i + 1 : i + context_length + 1] for i in ix])

        # 转换为 Tensor，设置类型为 long，并移动到指定 device
        x = torch.from_numpy(x_batch).long().to(device)
        y = torch.from_numpy(y_batch).long().to(device)

        yield x, y
