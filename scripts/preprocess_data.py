"""文本预处理脚本：

本脚本用于将原始文本数据转换为适用于神经网络训练的二进制格式（如 GPT-2/3/4 所用）。
它会读取每一行文本，使用指定的 BPE 分词器（由 vocab.json 和 merges.txt 定义）进行分词编码，
然后将所有分词后的 token ids 保存为高效的 uint16 二进制文件（.bin）。

用法示例：
python scripts/preprocess_data.py --input data/raw.txt --output data/tokenized.bin --vocab tests/fixtures/gpt2_vocab.json --merges tests/fixtures/gpt2_merges.txt

参数说明:
--input   指定原始文本文件路径，每行为一个样本
--output  指定输出的二进制 token id 文件路径
--vocab   BPE 分词器的 vocab 文件路径
--merges  BPE 分词器的 merges 文件路径

输出文件可被训练脚本直接高效加载。

"""

import os
import numpy as np
import argparse
from tqdm import tqdm
from cs336_basics.bpe_tokenizer import get_tokenizer_from_vocab_merges_path


def preprocess(input_path, output_path, vocab_path, merges_path, special_tokens=None):
    print(f"Loading tokenizer from {vocab_path} and {merges_path}...")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=vocab_path, merges_path=merges_path, special_tokens=special_tokens
    )

    print(f"Processing {input_path}...")
    all_token_ids = []

    # 获取文件总行数用于进度条（可选，大文件可能慢，可以跳过或者用估计值）
    # 这里我们直接按行读取并处理
    with open(input_path, "r", encoding="utf-8") as f:
        # 使用 tqdm 显示进度
        for line in tqdm(f, desc="Tokenizing"):
            token_ids = tokenizer.encode(line)
            all_token_ids.extend(token_ids)

    print(f"Saving to {output_path}...")
    # 转换为 uint16 并保存
    # 注意：如果 vocab_size > 65535，需要使用 uint32
    token_ids_np = np.array(all_token_ids, dtype=np.uint16)
    token_ids_np.tofile(output_path)
    print(f"Done! Saved {len(token_ids_np)} tokens.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess text data for training.")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to raw text file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save binary file"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="tests/fixtures/gpt2_vocab.json",
        help="Path to vocab file",
    )
    parser.add_argument(
        "--merges",
        type=str,
        default="tests/fixtures/gpt2_merges.txt",
        help="Path to merges file",
    )
    parser.add_argument(
        "--special_tokens", type=str, nargs="*", default=None, help="Special tokens"
    )

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    preprocess(
        args.input,
        args.output,
        args.vocab,
        args.merges,
        special_tokens=args.special_tokens,
    )
