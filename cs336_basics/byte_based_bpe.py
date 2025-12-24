import os
from typing import BinaryIO, Tuple, Set, Dict, List
import regex as re
from collections import Counter, defaultdict
from itertools import chain
from tqdm import tqdm
import multiprocessing
import json
from functools import lru_cache

# Byte-level regex pattern
PAT_BYTES = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def get_pairs(word):
    """Return set of symbol pairs in a word."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class ByteBPETrainer:
    """Byte-level Pair Encoding Trainer"""
    
    def __init__(self, vocab_size: int, special_tokens: List[str] = None):
        """
        初始化Byte-level BPE训练器
        
        Args:
            vocab_size: 目标词汇表大小
            special_tokens: 特殊token列表，默认包含 <|endoftext|>
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<|endoftext|>"]
        self.special_tokens_bytes = [token.encode('utf-8') for token in self.special_tokens]
        self.vocab = {}
        self.merge_rules = []
        
        # Byte-level预分词正则表达式
        self.PAT_BYTES = PAT_BYTES
    
    def get_base_vocab_bytes(self) -> List[bytes]:
        """
        获取基础256个单字节词汇表
        Returns:
            包含256个单字节的列表 [b'\x00', b'\x01', ..., b'\xff']
        """
        return [bytes([i]) for i in range(256)]
    
    def find_chunk_boundaries(self, file: BinaryIO, num_chunks: int, delimiter: bytes) -> List[int]:
        """
        找到文件中的分块边界，确保不会在delimiter中间分割
        
        Args:
            file: 文件对象
            num_chunks: 期望的分块数量
            delimiter: 特殊token的字节表示
        
        Returns:
            边界位置列表
        """
        file.seek(0, 2)  # 移动到文件末尾
        file_size = file.tell()
        file.seek(0)  # 重置到开头
        
        if num_chunks <= 1:
            return [0, file_size]
        
        chunk_size = file_size // num_chunks
        boundaries = [0]
        
        for i in range(1, num_chunks):
            target_pos = i * chunk_size
            file.seek(target_pos)
            
            # 读取一些数据以查找换行或delimiter边界
            buffer = file.read(min(1000, file_size - target_pos))
            
            # 寻找安全的分割点
            split_pos = target_pos
            newline_pos = buffer.find(b'\n')
            if newline_pos != -1:
                split_pos = target_pos + newline_pos + 1
            
            boundaries.append(split_pos)
        
        boundaries.append(file_size)
        return boundaries
    
    def process_chunk(self, chunk_info: Tuple[int, int, str]) -> Counter:
        """
        处理单个文件块
        
        Args:
            chunk_info: (start_pos, end_pos, file_path)
        
        Returns:
            该块的单词频率统计
        """
        start, end, file_path = chunk_info
        
        with open(file_path, 'rb') as f:
            f.seek(start)
            chunk = f.read(end - start)
        
        # 按special tokens分割
        delimiter_pattern = b"|".join(re.escape(token) for token in self.special_tokens_bytes)
        if delimiter_pattern:
            splited_chunks = re.split(delimiter_pattern, chunk)
        else:
            splited_chunks = [chunk]
        
        words = []
        for single_chunk in splited_chunks:
            if not single_chunk:
                continue
            
            # 使用字节级正则表达式进行分词
            for match in self.PAT_BYTES.finditer(single_chunk):
                words.append(match.group())
        
        return Counter(words)
    
    def train(self, input_path: str, num_processes: int = 8) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """
        训练Byte-level BPE模型 - 完全按照工作模拟实现
        
        Args:
            input_path: 训练文本文件路径
            num_processes: 并行处理进程数
            
        Returns:
            vocab: 词汇表字典 {token_id: token_bytes}
            merge_rules: 合并规则列表
        """
        # 初始化基础词汇表
        base_vocab = []
        
        # 添加special tokens（以bytes形式）
        for token_bytes in self.special_tokens_bytes:
            base_vocab.append(token_bytes)
        
        # 添加256个基础字节
        base_bytes = self.get_base_vocab_bytes()
        base_vocab.extend(base_bytes)
        
        # 并行处理文本块
        with open(input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, num_processes, 
                                                    self.special_tokens_bytes[0])
            
            word_freqs = Counter()
            
            # 并行处理
            with multiprocessing.Pool(processes=num_processes) as pool:
                tasks = [(start, end, input_path) 
                        for start, end in zip(boundaries[:-1], boundaries[1:])]
                results = pool.map(self.process_chunk, tasks)
            
            # 合并结果
            for chunk_word_freqs in results:
                word_freqs.update(chunk_word_freqs)
        
        # 构建初始词汇表（id -> bytes映射）
        vocab = dict(zip(range(len(base_vocab)), base_vocab))
        
        # 将每个word转换为字节列表（和模拟中一样）
        word_splits = {}
        for word in word_freqs:
            word_splits[word] = [bytes([b]) for b in word]  # 每个字节作为一个bytes对象
        
        def get_pairs_from_splits(word_splits, word_freqs):
            """从当前分割中统计所有相邻pair的频率"""
            pairs = defaultdict(int)
            for word, splits in word_splits.items():
                freq = word_freqs[word]
                for i in range(len(splits) - 1):
                    pair = (splits[i], splits[i+1])
                    pairs[pair] += freq
            return pairs
        
        def merge_pair_in_splits(word_splits, pair):
            """在分割中合并指定的pair"""
            new_word_splits = {}
            for word, splits in word_splits.items():
                new_splits = []
                i = 0
                while i < len(splits):
                    if i < len(splits) - 1 and splits[i] == pair[0] and splits[i+1] == pair[1]:
                        # 合并这两个元素
                        merged = splits[i] + splits[i+1]
                        new_splits.append(merged)
                        i += 2
                    else:
                        new_splits.append(splits[i])
                        i += 1
                new_word_splits[word] = new_splits
            return new_word_splits
        
        # 执行BPE合并
        merges = []
        
        pbar = tqdm(total=self.vocab_size - len(vocab), desc="Merging byte pairs")
        
        for i in range(self.vocab_size - len(vocab)):
            pairs = get_pairs_from_splits(word_splits, word_freqs)
            
            if not pairs:
                break
            
            # 找频率最高的pair
            max_freq = max(pairs.values())
            best_pairs = [pair for pair, freq in pairs.items() if freq == max_freq]
            
            # Tie-breaking: 选择字典序最大的pair，基于原始bytes
            # 根据CHANGELOG.md 0.1.1: "tiebreaking on this remapped unicode representation instead of the original bytes"
            # 我们应该基于原始bytes进行tie-breaking
            def get_bytes_key(pair):
                # 直接比较bytes对象
                return (pair[0], pair[1])
            
            # 选择原始bytes下字典序最大的pair
            best_pair = max(best_pairs, key=get_bytes_key)
            
            # 记录合并规则
            merges.append(best_pair)
            
            # 更新词汇表
            new_token = best_pair[0] + best_pair[1]
            vocab[len(vocab)] = new_token
            
            # 执行merge
            word_splits = merge_pair_in_splits(word_splits, best_pair)
            pbar.update(1)
        
        pbar.close()
        
        self.vocab = vocab
        self.merge_rules = merges
        
        return vocab, merges
    
    def get_vocab(self) -> Dict[int, bytes]:
        """获取训练后的词汇表"""
        return self.vocab
    
    def get_merge_rules(self) -> List[Tuple[bytes, bytes]]:
        """获取训练后的合并规则"""
        return self.merge_rules
    
    @staticmethod
    @lru_cache
    def bytes_to_unicode() -> Dict[int, str]:
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(n) for n in cs]))
    
    def save_vocab(self, vocab_path: str):
        byte_encoder = self.bytes_to_unicode()
        saved_vocab_dict = {}
        
        for token_id, token_bytes in self.vocab.items():
            token_str = ''.join(byte_encoder[b] for b in token_bytes)
            saved_vocab_dict[token_str] = token_id
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(saved_vocab_dict, f, ensure_ascii=False, indent=4)
    
    def save_merge_rules(self, rules_path: str):
        byte_encoder = self.bytes_to_unicode()
        
        with open(rules_path, 'w', encoding='utf-8') as f:
            for pair in self.merge_rules:
                str1 = ''.join(byte_encoder[b] for b in pair[0])
                str2 = ''.join(byte_encoder[b] for b in pair[1])
                f.write(f"{str1} {str2}\n")


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    训练Byte-level BPE模型的便捷函数
    
    Args:
        input_path: 训练文本文件路径
        vocab_size: 目标词汇表大小
        special_tokens: 特殊token列表
    
    Returns:
        vocab: 词汇表字典 {token_id: token_bytes}
        merge_rules: 合并规则列表
    """
    trainer = ByteBPETrainer(vocab_size, special_tokens)
    return trainer.train(input_path)


if __name__ == "__main__":
    
    trainer = ByteBPETrainer(
        vocab_size=32000,
        special_tokens=["<|endoftext|>"]
    )
    
    vocab_save_path = "data/bpe_result/owt_bpe_vocab.json"
    merged_rules_save_path = "data/bpe_result/owt_bpe_merges.txt"
    # test_file_path = "tests/fixtures/corpus.en"
    test_file_path = "data/owt_train.txt"
    
    vocab_dict, merged_rules = trainer.train(test_file_path, num_processes=8)
    
    print(f"训练完成！词汇表大小: {len(vocab_dict)}")
    print(f"合并规则数量: {len(merged_rules)}")
    
    trainer.save_vocab(vocab_save_path)
    trainer.save_merge_rules(merged_rules_save_path)