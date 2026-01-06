import os
import torch
from dataclasses import dataclass, field
from cs336_basics.transformer_lm import Transformer_LM
from cs336_basics.utils import load_checkpoint
from cs336_basics.bpe_tokenizer import get_tokenizer_from_vocab_merges_path
from cs336_basics.softmax import softmax_function, top_p_sampling


def prompts_to_token_ids(prompts: list[str]):
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path="tests/fixtures/gpt2_vocab.json",
        merges_path="tests/fixtures/gpt2_merges.txt",
    )
    return [tokenizer.encode(prompt) for prompt in prompts]


def decode_token_ids(token_ids: list[int]):
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path="tests/fixtures/gpt2_vocab.json",
        merges_path="tests/fixtures/gpt2_merges.txt",
    )
    return tokenizer.decode(token_ids)


user_input = "Please write a story about a cat: "


@dataclass
class InferenceConfig:
    # 模型超参数
    vocab_size: int = 50257  # 根据实际 tokenizer 调整
    context_length: int = 512  # 推荐: 128-512 (对于 TinyStories, 256-512 效果更好)
    d_model: int = 512  # 推荐: 128-768 (需被 num_heads 整除)
    num_layers: int = 4  # 推荐: 4-12 (层数多利于理解复杂语法)
    num_heads: int = (
        16  # 推荐: 4-12 (每个 head 的维度 d_model/num_heads 建议在 32-128 之间)
    )
    d_ff: int = 1344  # 推荐: 4 * d_model (标准 Transformer 比例)
    rope_theta: float = 10000.0  # 常用值: 10000.0 (外推需求大时可调大)

    temperature: float = 1.0
    top_p: float = 0.9
    max_new_tokens: int = 50
    stop_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>"])

    compile: bool = True  # 推理也开启编译
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型路径
    model_path: str = "checkpoints/best_model.pt"


def main_infer(config: InferenceConfig):

    device = torch.device(config.device)
    print(f"Using device: {device}")

    # 提前初始化 Tokenizer，避免在循环中重复创建
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path="tests/fixtures/gpt2_vocab.json",
        merges_path="tests/fixtures/gpt2_merges.txt",
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
    model.to(device)

    if os.path.exists(config.model_path):
        load_checkpoint(config.model_path, model)
        print(f"Loaded model from {config.model_path}")
    else:
        print(
            f"Warning: Model path {config.model_path} does not exist. Using uninitialized model."
        )

    model.eval()

    # 编译模型
    if config.compile:
        print("Compiling model for inference...")
        model = torch.compile(model, mode="reduce-overhead")

    # 编码 prompt
    token_ids_list = tokenizer.encode(user_input)
    token_ids = torch.tensor([token_ids_list], device=device, dtype=torch.long)

    print("Generating...")

    # 将停止词转换为 ID 集合
    stop_token_ids = set()
    for stop_str in config.stop_tokens:
        ids = tokenizer.encode(stop_str)
        if ids:
            stop_token_ids.add(ids[0])

    with torch.inference_mode():
        for _ in range(config.max_new_tokens):
            # 裁剪输入以符合 context_length
            input_ids = token_ids[:, -config.context_length :]

            # 模型推理
            logits = model(input_ids)
            raw_output = logits[:, -1, :]

            # 1. 带 temperature 的 softmax
            softmax_output = softmax_function(raw_output / config.temperature, dim=-1)

            # 2. 带 top-p 的采样
            top_p_output = top_p_sampling(softmax_output, config.top_p)

            # 3. 采样下一个 token
            probs = torch.softmax(top_p_output, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 4. 将下一个 token 添加到 token_ids 中
            token_ids = torch.cat([token_ids, next_token], dim=-1)

            # 如果生成了任何停止符，则提前停止
            if next_token.item() in stop_token_ids:
                print(f"Generated stop token. Stopping.")
                break

    # 5. 返回生成的文本
    return tokenizer.decode(token_ids[0].tolist())


if __name__ == "__main__":
    generated_text = main_infer(InferenceConfig())
    print(f"\nGenerated text:\n{generated_text}")
