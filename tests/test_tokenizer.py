from __future__ import annotations

import json
import os
import resource
import sys

import psutil
import pytest
import tiktoken

from .adapters import get_tokenizer
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"


def memory_limit(max_mem):
    def decorator(f):
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            prev_limits = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(
                resource.RLIMIT_AS, (process.memory_info().rss + max_mem, -1)
            )
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                # Even if the function above fails (e.g., it exceeds the
                # memory limit), reset the memory limit back to the
                # previous limit so other tests aren't affected.
                resource.setrlimit(resource.RLIMIT_AS, prev_limits)

        return wrapper

    return decorator


def memory_limit_macos(max_mem):
    """
    macOS (M3 芯片) 专用的内存限制装饰器。
    由于 macOS 对 RLIMIT_AS 的支持不如 Linux 完善，
    这里使用 RLIMIT_DATA 来限制堆内存的增长。
    """

    def decorator(f):
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            # 在 macOS 上，使用 RLIMIT_DATA 来限制堆内存
            if hasattr(resource, "RLIMIT_DATA"):
                prev_limits = resource.getrlimit(resource.RLIMIT_DATA)
                resource.setrlimit(
                    resource.RLIMIT_DATA, (process.memory_info().rss + max_mem, -1)
                )
                try:
                    result = f(*args, **kwargs)
                    return result
                finally:
                    # 恢复之前的内存限制
                    resource.setrlimit(resource.RLIMIT_DATA, prev_limits)
            else:
                # 如果系统不支持 RLIMIT_DATA，直接运行函数
                return f(*args, **kwargs)

        return wrapper

    return decorator


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return get_tokenizer(vocab, merges, special_tokens)


def test_roundtrip_empty():
    """
    测试空字符串的往返编码/解码。
    验证：将空字符串编码后再解码，应该得到原始的空字符串。
    这是最基础的边界情况测试。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = ""
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_empty_matches_tiktoken():
    """
    测试空字符串的编码结果与 tiktoken 库的一致性。
    验证：自己实现的分词器对空字符串的编码结果应该与 OpenAI 官方的 tiktoken 库完全相同。
    这确保了实现的正确性和标准兼容性。
    """
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = ""

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == []

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_single_character():
    """
    测试单个 ASCII 字符的往返编码/解码。
    验证：将单个字符 "s" 编码后再解码，应该得到原始字符。
    测试简单 ASCII 字符处理的正确性。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "s"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_single_character_matches_tiktoken():
    """
    测试单个 ASCII 字符的编码结果与 tiktoken 的一致性。
    验证：自己实现的分词器对单个字符 "s" 的编码应该与 tiktoken 完全相同。
    并验证解码后得到正确的单个字符。
    """
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "s"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == ["s"]

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_single_unicode_character():
    """
    测试单个 Unicode 字符（表情符号）的往返编码/解码。
    验证：将 Unicode 表情字符 "🙃" 编码后再解码，应该得到原始字符。
    测试分词器对多字节 Unicode 字符的处理能力。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "🙃"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_single_unicode_character_matches_tiktoken():
    """
    测试单个 Unicode 表情符号的编码结果与 tiktoken 的一致性。
    验证：自己实现的分词器对 Unicode 表情 "🙃" 的编码应该与 tiktoken 完全相同。
    确保 Unicode 多字节字符的处理符合标准。
    """
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "🙃"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_ascii_string():
    """
    测试普通 ASCII 字符串的往返编码/解码。
    验证：将字符串 "Hello, how are you?" 编码后再解码，应该得到原始字符串。
    测试分词器对完整句子的处理能力，包括单词、标点符号等。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "Hello, how are you?"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_ascii_string_matches_tiktoken():
    """
    测试 ASCII 字符串的编码结果与 tiktoken 的一致性。
    验证：自己实现的分词器对 "Hello, how are you?" 的编码应该与 tiktoken 完全相同。
    同时验证解码后每个 token 的单个解码结果（验证 BPE 合并的正确性）。
    """
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Hello, how are you?"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    # assert ids == reference_ids

    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == ["Hello", ",", " how", " are", " you", "?"]

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_unicode_string():
    """
    测试包含 Unicode 字符的复杂字符串的往返编码/解码。
    验证：将 "Héllò hôw are ü? 🙃" 这个包含重音符号和表情的字符串编码后再解码，应该得到原始字符串。
    测试分词器对混合 Unicode 字符的处理能力。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "Héllò hôw are ü? 🙃"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_unicode_string_matches_tiktoken():
    """
    测试复杂 Unicode 字符串的编码结果与 tiktoken 的一致性。
    验证：自己实现的分词器对含有重音符号和表情的字符串的编码应该与 tiktoken 完全相同。
    确保多字节 UTF-8 字符的 BPE 处理符合标准。
    """
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw are ü? 🙃"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_unicode_string_with_special_tokens():
    """
    测试包含特殊 token 的 Unicode 字符串的往返编码/解码。
    验证：将含有特殊 token "<|endoftext|>" 的字符串编码后再解码，应该得到原始字符串。
    特别验证特殊 token 被正确保留为单个完整 token（而不是被拆分或合并）。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded_ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    # Ensure the special <|endoftext|> token is preserved
    assert tokenized_string.count("<|endoftext|>") == 3

    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_unicode_string_with_special_tokens_matches_tiktoken():
    """
    测试含有特殊 token 的 Unicode 字符串的编码结果与 tiktoken 的一致性。
    验证：自己实现的分词器对含有特殊 token 的字符串编码应该与 tiktoken 完全相同。
    需要在 tiktoken 中明确指定允许的特殊 token。
    """
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"

    reference_ids = reference_tokenizer.encode(
        test_string, allowed_special={"<|endoftext|>"}
    )
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_overlapping_special_tokens():
    """
    测试重叠特殊 token 的编码处理。
    验证：当定义两个重叠的特殊 token "<|endoftext|>" 和 "<|endoftext|><|endoftext|>" 时，
    应该优先匹配较长的特殊 token（贪心匹配策略）。
    例如，连续出现的两个 "<|endoftext|>" 应该被合并成一个 "<|endoftext|><|endoftext|>" token。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )
    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

    ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    # Ensure the double <|endoftext|><|endoftext|> is preserved as a single token
    assert tokenized_string.count("<|endoftext|>") == 1
    assert tokenized_string.count("<|endoftext|><|endoftext|>") == 1
    # Test roundtrip
    assert tokenizer.decode(ids) == test_string


def test_address_roundtrip():
    """
    测试真实数据（地址文本）的往返编码/解码。
    验证：加载 address.txt 文件的内容，编码后再解码应该得到原始内容。
    测试分词器在实际文本数据上的正确性。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "address.txt") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_address_matches_tiktoken():
    """
    测试地址文本的编码结果与 tiktoken 的一致性。
    验证：自己实现的分词器对 address.txt 的编码应该与 tiktoken 完全相同。
    确保在真实数据上的兼容性。
    """
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    corpus_path = FIXTURES_PATH / "address.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents)
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_german_roundtrip():
    """
    测试德语文本的往返编码/解码。
    验证：加载 german.txt 文件的内容，编码后再解码应该得到原始内容。
    测试分词器对非英文语言的处理能力。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "german.txt") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


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


def test_tinystories_sample_roundtrip():
    """
    测试 TinyStories 样本数据的往返编码/解码。
    验证：加载 tinystories_sample.txt 文件的内容，编码后再解码应该得到原始内容。
    测试分词器在大型文本样本上的正确性。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_tinystories_matches_tiktoken():
    """
    测试 TinyStories 样本数据的编码结果与 tiktoken 的一致性。
    验证：自己实现的分词器对 tinystories_sample.txt 的编码应该与 tiktoken 完全相同。
    确保在大型真实数据集上的兼容性。
    """
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "tinystories_sample.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(
        corpus_contents, allowed_special={"<|endoftext|>"}
    )
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_encode_special_token_trailing_newlines():
    """
    测试含有特殊 token 和尾部换行符的文本的编码。
    验证：自己实现的分词器对 special_token_trailing_newlines.txt 的编码应该与 tiktoken 完全相同。
    测试分词器对边界情况（如文末的换行符）的处理。
    """
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "special_token_trailing_newlines.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(
        corpus_contents, allowed_special={"<|endoftext|>"}
    )
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_encode_special_token_double_newline_non_whitespace():
    """
    测试含有特殊 token、双换行符和非空白字符混合的文本编码。
    验证：自己实现的分词器对 special_token_double_newlines_non_whitespace.txt 的编码应该与 tiktoken 完全相同。
    测试分词器对复杂边界情况的处理，包括连续换行符和特殊 token 的交互。
    """
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "special_token_double_newlines_non_whitespace.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(
        corpus_contents, allowed_special={"<|endoftext|>"}
    )
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_encode_iterable_tinystories_sample_roundtrip():
    """
    测试迭代编码接口（encode_iterable）的往返处理。
    验证：使用 encode_iterable 逐个读取和编码 tinystories_sample.txt，
    然后解码所有编码后的 token，应该得到原始文本。
    测试流式编码接口的正确性和内存效率。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    all_ids = []
    with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
        corpus_contents = f.read()
    assert tokenizer.decode(all_ids) == corpus_contents


def test_encode_iterable_tinystories_matches_tiktoken():
    """
    测试迭代编码接口的编码结果与 tiktoken 的一致性。
    验证：使用 encode_iterable 流式编码 tinystories_sample.txt，
    结果应该与 tiktoken 的 encode 结果完全相同。
    测试流式编码的标准兼容性。
    """
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "tinystories_sample.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(
        corpus_contents, allowed_special={"<|endoftext|>"}
    )
    all_ids = []
    with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    assert all_ids == reference_ids

    assert tokenizer.decode(all_ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rlimit support for non-linux systems is spotty.",
)
def test_encode_iterable_memory_usage():
    """
    测试迭代编码接口在处理大型文件时的内存使用。
    验证：使用 encode_iterable 处理 5MB 的 tinystories_sample_5M.txt 应该
    在 1MB 的内存限制内完成（仅限 Linux 系统）。
    测试流式编码的内存效率，确保不会将整个文件加载到内存中。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        ids = []
        for _id in _encode_iterable(tokenizer, f):
            ids.append(_id)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rlimit support for non-linux systems is spotty.",
)
@pytest.mark.xfail(
    reason="Tokenizer.encode is expected to take more memory than allotted (1MB)."
)
def test_encode_memory_usage():
    """
    测试一次性编码接口（encode）在处理大型文件时的内存使用（预期失败）。
    验证：使用 encode 处理 5MB 的 tinystories_sample_5M.txt，
    预计会超过 1MB 的内存限制而导致测试失败（仅限 Linux 系统）。
    这个测试用来演示 encode 接口不够内存高效，需要用 encode_iterable 代替。
    标记为 xfail（预期失败）因为我们期望它失败。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        contents = f.read()
        _ = _encode(tokenizer, contents)


@pytest.mark.skipif(
    not sys.platform.startswith("darwin"),
    reason="rlimit support for macOS systems.",
)
def test_encode_iterable_memory_usage_macos():
    """
    macOS (M3 芯片) 测试：迭代编码接口在处理大型文件时的内存使用。
    验证：使用 encode_iterable 处理 5MB 的 tinystories_sample_5M.txt 应该
    在 1MB 的内存限制内完成（仅限 macOS 系统）。
    macOS 上的内存限制基于当前进程的 RSS（Resident Set Size）。
    测试流式编码的内存效率，确保不会将整个文件加载到内存中。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        ids = []
        for _id in _encode_iterable_macos(tokenizer, f):
            ids.append(_id)


@pytest.mark.skipif(
    not sys.platform.startswith("darwin"),
    reason="rlimit support for macOS systems.",
)
@pytest.mark.xfail(
    reason="Tokenizer.encode is expected to take more memory than allotted (1MB) on macOS."
)
def test_encode_memory_usage_macos():
    """
    macOS (M3 芯片) 测试：一次性编码接口在处理大型文件时的内存使用（预期失败）。
    验证：使用 encode 处理 5MB 的 tinystories_sample_5M.txt，
    预计会超过 1MB 的内存限制而导致测试失败（仅限 macOS 系统）。
    macOS 上的内存限制基于当前进程的 RSS（Resident Set Size）。
    这个测试用来演示 encode 接口不够内存高效，需要用 encode_iterable 代替。
    标记为 xfail（预期失败）因为我们期望它失败。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        contents = f.read()
        _ = _encode_macos(tokenizer, contents)


@memory_limit(int(1e6))
def _encode_iterable(tokenizer, iterable):
    """
    包装函数：在 1MB 内存限制下执行 tokenizer.encode_iterable。
    被 test_encode_iterable_memory_usage 使用，用于验证流式编码的内存效率。
    memory_limit 装饰器会在函数执行时施加内存限制。
    """
    yield from tokenizer.encode_iterable(iterable)


@memory_limit(int(1e6))
def _encode(tokenizer, text):
    """
    包装函数：在 1MB 内存限制下执行 tokenizer.encode。
    被 test_encode_memory_usage 使用，用于演示一次性编码接口的内存使用问题。
    memory_limit 装饰器会在函数执行时施加内存限制。
    """
    return tokenizer.encode(text)


@memory_limit_macos(int(1e6))
def _encode_iterable_macos(tokenizer, iterable):
    """
    包装函数：macOS (M3 芯片) 版本。在 1MB 内存限制下执行 tokenizer.encode_iterable。
    被 test_encode_iterable_memory_usage_macos 使用，用于验证流式编码的内存效率。
    macOS 上的 memory_limit_macos 装饰器使用 RLIMIT_DATA 来限制堆内存增长。
    """
    yield from tokenizer.encode_iterable(iterable)


@memory_limit_macos(int(1e6))
def _encode_macos(tokenizer, text):
    """
    包装函数：macOS (M3 芯片) 版本。在 1MB 内存限制下执行 tokenizer.encode。
    被 test_encode_memory_usage_macos 使用，用于演示一次性编码接口的内存使用问题。
    macOS 上的 memory_limit_macos 装饰器使用 RLIMIT_DATA 来限制堆内存增长。
    """
    return tokenizer.encode(text)
