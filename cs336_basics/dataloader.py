import numpy as np
import torch
from typing import Generator
import numpy.typing as npt

def simple_dataloader(dataset: npt.NDArray, 
                      batch_size: int, 
                      context_length: int,
                      random_start: bool = True) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    根据 tests/adapters.py 中的需求实现的简单数据加载器。
    
    该生成器会随机采样起始位置来构建批次，这符合 Language Modeling 的训练需求。
    
    Args:
        dataset: 1D numpy 数组，包含 integer token IDs。
        batch_size: 每批采样的序列数量。
        context_length: 每个序列的上下文长度。
        
    Yields:
        (inputs, labels) 元组，形状均为 (batch_size, context_length) 的 torch.LongTensor。
    """
    n = len(dataset)
    # 为了保证 y = x + 1 且不越界，最大的起始索引 i 需满足 i + context_length < n
    # 即 i <= n - context_length - 1
    # np.random.randint 的 high 是 open interval，所以传入 n - context_length
    max_start_idx = n - context_length
    
    while True:
        # 随机产生 batch_size 个起始位置
        if random_start:
            ix = np.random.randint(0, max_start_idx, size=batch_size)
        else:
            ix = np.arange(0, max_start_idx, step=batch_size)
        
        # 构造输入 x 和标签 y
        # x 为 [start, start + context_length)
        # y 为 [start + 1, start + context_length + 1)
        x_batch = np.stack([dataset[i : i + context_length] for i in ix])
        y_batch = np.stack([dataset[i + 1 : i + context_length + 1] for i in ix])
        
        yield torch.from_numpy(x_batch).long(), torch.from_numpy(y_batch).long()
