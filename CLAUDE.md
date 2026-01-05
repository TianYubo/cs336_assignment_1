# CLAUDE.md - 项目开发指南

## 1. 项目简介与目标
本项目是斯坦福 CS336 课程的作业实现，旨在从零开始构建一个 Transformer 语言模型。主要目标包括：
- 实现字节级 BPE（Byte-Pair Encoding）分词器。
- 手写 Transformer 的各个组件（Attention, RMSNorm, SwiGLU, RoPE 等）。
- 集成并训练一个完整的 Transformer LM。
- 在 TinyStories 等数据集上进行模型训练与推理。

## 2. 技术栈与版本
- **语言**: Python >= 3.11
- **包管理**: [uv](https://github.com/astral-sh/uv) (推荐) 或 pip
- **深度学习**: PyTorch ~= 2.6.0
- **矩阵操作**: NumPy, einops, einx
- **类型检查**: jaxtyping (配合 torch/numpy 使用)
- **代码规范**: Ruff (配置见 `pyproject.toml`)
- **监控**: TensorBoard, WandB

## 3. 开发与运行命令
### 环境准备
```bash
uv sync  # 安装所有依赖
```

### 数据下载与预处理
```bash
# 下载数据 (需手动执行或运行对应脚本)
mkdir -p data && cd data
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
cd ..

# 文本预处理为二进制 token IDs
uv run python scripts/preprocess_data.py --input data/TinyStoriesV2-GPT4-train.txt --output data/TinyStoriesV2-GPT4-train.bin
```

### 训练与生成
```bash
# 开始训练
uv run python scripts/train.py

# 文本生成
uv run python scripts/text_generation.py --prompt "Once upon a time"
```

### 测试命令
```bash
uv run pytest              # 运行所有测试
uv run pytest tests/test_model.py  # 运行特定测试
```

## 4. 目录结构
- `cs336_basics/`: 核心代码库，包含模型组件、分词器、优化器等。
- `scripts/`: 任务脚本，包括数据预处理、训练、推理。
- `tests/`: 单元测试，包含 `adapters.py`（用于将你的实现对接至标准测试）。
- `data/`: 存放原始文本和预处理后的二进制数据。
- `checkpoints/`: 训练好的模型权重。
- `logs/`: TensorBoard 日志。

## 5. 编码规范
- **类型提示**: 强制使用 `jaxtyping` 和 `torch.Tensor` 进行类型标注，例如 `Float[Tensor, "batch seq d_model"]`。
- **代码风格**: 遵循 `ruff` 配置。行宽限制 120。
- **模块化**: 复杂的算子需继承 `torch.nn.Module` 并实现在 `cs336_basics/` 对应的模块中。
- **错误处理**: 核心组件应在输入维度不匹配时抛出清晰的 `AssertionError` 或 `ValueError`。

## 6. 变更规则
- **测试优先**: 修改 `cs336_basics/` 下的逻辑后，必须运行 `pytest` 确保通过。
- **禁止修改测试用例**: `tests/` 下除了 `adapters.py` 之外的文件原则上禁止修改，以保证评估基准一致。
- **适配器模式**: 若实现与测试接口不一致，应在 `tests/adapters.py` 中进行封装，不要直接改动核心代码。

## 7. 常见任务 SOP
### 加新模型组件
1. 在 `cs336_basics/` 下创建新的 `.py` 文件或在现有文件中添加类。
2. 在 `tests/adapters.py` 中添加或更新对应的包装函数。
3. 运行相应的单元测试。

### 调整训练配置
1. 修改 `scripts/train.py` 中的 `TrainingConfig` 数据类。
2. 确保 `data/` 下已生成对应的 `.bin` 文件。

## 8. 已知问题与排查
- **显存不足 (OOM)**: 调小 `TrainingConfig` 中的 `batch_size` 或 `d_model`。
- **模型未收敛**: 检查 `SimpleAdamW` 的实现以及 `lr_schedule` 的热身步数。
- **测试失败**: 优先检查 `tests/adapters.py` 是否正确导入并调用了你的最新实现。
- **数据加载错误**: 确保已运行 `preprocess_data.py` 将文本转换为 `uint16` 格式。

