#!/usr/bin/env python3
"""
测试 test_german_matches_tiktoken 用例
"""

import json
import os
from pathlib import Path

import tiktoken

from cs336_basics.bpe_tokenizer import get_tokenizer_from_vocab_merges_path
from cs336_basics.common import gpt2_bytes_to_unicode

FIXTURES_PATH = Path("tests/fixtures")
VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"


def test_german_matches_tiktoken():
    """
    测试德语文本的编码结果与 tiktoken 的一致性。
    验证：自己实现的分词器对 german.txt 的编码应该与 tiktoken 完全相同。
    确保对多语言的支持与标准兼容。
    """
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    corpus_path = FIXTURES_PATH / "german.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents)
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


if __name__ == "__main__":
    print("=" * 80)
    print("测试 test_german_matches_tiktoken")
    print("=" * 80 + "\n")

    try:
        test_german_matches_tiktoken()
        print("✓ 测试通过！")
    except AssertionError as e:
        print(f"✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback

        traceback.print_exc()
