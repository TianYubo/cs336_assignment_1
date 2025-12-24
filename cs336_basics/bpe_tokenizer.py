import os
import json
from cs336_basics.common import gpt2_bytes_to_unicode
from typing import Iterable, Iterator
import re
import regex


class BPETokenizer:
    """
    A simple BPE tokenizer that uses a given vocabulary and merge rules to
    tokenize and detokenize text.
    """

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.byte_encoder = gpt2_bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = {tuple(merge): rank for rank, merge in enumerate(self.merges)}
        self.bpe_cache: dict[bytes, tuple[bytes, ...]] = {}
        self._bpe_pattern = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            flags=regex.UNICODE,
        )

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Load tokenizer from vocab and merges files.
        """
        vocab = {}
        merges = []
        if vocab_filepath:
            with open(vocab_filepath, "r", encoding="utf-8") as f:
                for line in f:
                    token, idx = line.strip().split()
                    vocab[token] = int(idx)
        if merges_filepath:
            with open(merges_filepath, "r", encoding="utf-8") as f:
                for line in f:
                    merges.append(tuple(line.strip().split()))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        使用 BPE merges 对文本分词，保持与 GPT-2 / tiktoken 的兼容性。

        Args:
            text: 输入的文本字符串

        Returns:
            list[int]: token_id 列表
        """

        # ====================================================================
        # 步骤 1：建立特殊 token 到 token_id 的映射
        # ====================================================================
        special_token_to_id: dict[str, int] = {}
        if self.special_tokens:
            for special_token in self.special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes in self.inv_vocab:
                    special_token_to_id[special_token] = self.inv_vocab[
                        special_token_bytes
                    ]

        token_ids: list[int] = []

        # ====================================================================
        # 步骤 2：分割文本为特殊 token 和普通部分
        # ====================================================================
        if special_token_to_id:
            special_tokens_list = sorted(
                special_token_to_id.keys(), key=len, reverse=True
            )
            escaped_tokens = [re.escape(token) for token in special_tokens_list]
            pattern = "(" + "|".join(escaped_tokens) + ")"
            segments = re.split(pattern, text)
        else:
            segments = [text]

        # ====================================================================
        # 步骤 3：处理每个分割部分
        # ====================================================================
        for segment in segments:
            if not segment:
                continue

            if segment in special_token_to_id:
                token_ids.append(special_token_to_id[segment])
                continue

            for token in self._bpe_pattern.findall(segment):
                token_bytes = token.encode("utf-8")
                for bpe_piece in self._apply_bpe(token_bytes):
                    token_id = self.inv_vocab.get(bpe_piece)
                    if token_id is not None:
                        token_ids.append(token_id)
                        continue

                    # 理论上应该总能命中 vocab；若未命中，退化为逐字节编码
                    for byte_val in bpe_piece:
                        single_byte = bytes([byte_val])
                        single_id = self.inv_vocab.get(single_byte)
                        if single_id is None:
                            raise KeyError(
                                f"Unknown byte sequence encountered: {bpe_piece!r}"
                            )
                        token_ids.append(single_id)

        return token_ids

    @staticmethod
    def _get_pairs(word: list[bytes]) -> set[tuple[bytes, bytes]]:
        """
        获取当前分词结果中所有相邻的 bytes 对。
        """
        pairs = set()
        if not word:
            return pairs
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs

    def _apply_bpe(self, token_bytes: bytes) -> tuple[bytes, ...]:
        """
        对单个 token 的原始字节序列执行 BPE merges，返回拆分后的多个子词 bytes。
        """
        if not token_bytes:
            return ()

        if token_bytes in self.bpe_cache:
            return self.bpe_cache[token_bytes]

        word = [bytes([b]) for b in token_bytes]
        pairs = self._get_pairs(word)

        if not pairs:
            result = (token_bytes,)
            self.bpe_cache[token_bytes] = result
            return result

        while True:
            best_pair = None
            best_rank = None
            for pair in pairs:
                rank = self.bpe_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            first, second = best_pair
            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)

        result = tuple(word)
        self.bpe_cache[token_bytes] = result
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:  # line 是一行文本
            # 编码这一行，得到多个 token IDs
            token_ids = self.encode(line)  # 例如：[123, 456, 789]
            # 逐个 yield 每个 token ID
            for token_id in token_ids:
                yield token_id  # 每次返回一个整数

    def decode(self, token_ids):
        """
        将 token_ids 列表解码回文本字符串。

        Args:
            token_ids: token_id 列表

        Returns:
            str: 解码后的文本
        """
        decoded_parts = []
        byte_buffer = bytearray()

        def flush_buffer():
            # 将累计的字节序列解码为字符串，并且清空缓冲区
            if byte_buffer:
                decoded_parts.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token_id in token_ids:
            if token_id in self.vocab:
                token_value = self.vocab[token_id]
                if isinstance(token_value, bytes):
                    # 只累计，不立即解码（因为可能是多字节字符的一部分）
                    byte_buffer.extend(token_value)
                else:
                    # 非 bytes 类型，先 flush buffer 结束之前的字节序列，再添加该部分
                    flush_buffer()
                    decoded_parts.append(str(token_value))
            else:
                # 未知 token_id，先 flush buffer，再添加占位符
                flush_buffer()
                decoded_parts.append("<unk>")

        # 循环结束后，flush 剩余的字节序列
        flush_buffer()

        return "".join(decoded_parts)


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
    return BPETokenizer(vocab, merges, special_tokens)


if __name__ == "__main__":

    vocab_path = "tests/fixtures/gpt2_vocab.json"
    merges_path = "tests/fixtures/gpt2_merges.txt"

    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=vocab_path,
        merges_path=merges_path,
    )

    sample = "Hello, world!"
    print(tokenizer.encode(sample))
