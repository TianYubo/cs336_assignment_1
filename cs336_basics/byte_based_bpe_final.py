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
    """Byte-level Pair Encoding Trainer - Exact Reference Implementation"""
    
    def __init__(self, vocab_size: int, special_tokens: List[str] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<|endoftext|>"]
        self.special_tokens_bytes = [token.encode('utf-8') for token in self.special_tokens]
        self.vocab = {}
        self.merge_rules = []
        self.PAT_BYTES = PAT_BYTES
    
    def get_base_vocab_bytes(self) -> List[bytes]:
        return [bytes([i]) for i in range(256)]
    
    def find_chunk_boundaries(self, file: BinaryIO, desired_num_chunks: int, 
                              split_special_token: bytes) -> List[int]:
        assert isinstance(split_special_token, bytes)
        
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)
            while True:
                mini_chunk = file.read(mini_chunk_size)
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        return sorted(set(chunk_boundaries))
    
    def get_word_frequency_in_chunk_bytes(self, chunk: bytes) -> Dict[Tuple[bytes, ...], int]:
        delimiter_pattern = b"|".join(re.escape(token) for token in self.special_tokens_bytes)
        if delimiter_pattern:
            splited_chunks = re.split(delimiter_pattern, chunk)
        else:
            splited_chunks = [chunk]
        
        words = []
        for single_chunk in splited_chunks:
            if not single_chunk:
                continue
            for match in self.PAT_BYTES.finditer(single_chunk):
                words.append(match.group())
                
        word_freq_counter = Counter(words)
        
        word_byte_lists = {}
        for word, freq in word_freq_counter.items():
            byte_tuple = tuple(bytes([b]) for b in word)
            word_byte_lists[byte_tuple] = freq
        
        return word_byte_lists
    
    def process_chunk(self, args: Tuple[int, int, str]) -> Dict[Tuple[bytes, ...], int]:
        start, end, input_path = args
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start)
        return self.get_word_frequency_in_chunk_bytes(chunk)
    
    def train(self, input_path: str, num_processes: int = 8) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        # 初始化词汇表
        base_vocab = []
        for token_bytes in self.special_tokens_bytes:
            base_vocab.append(token_bytes)
        base_bytes = self.get_base_vocab_bytes()
        base_vocab.extend(base_bytes)
        
        # 处理文本
        with open(input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, num_processes, 
                                                    self.special_tokens_bytes[0])
            
            word_freqs = Counter()
            
            with multiprocessing.Pool(processes=num_processes) as pool:
                tasks = [(start, end, input_path) 
                        for start, end in zip(boundaries[:-1], boundaries[1:])]
                results = pool.map(self.process_chunk, tasks)
            
            for chunk_word_freqs in results:
                word_freqs.update(chunk_word_freqs)
        
        vocab = dict(zip(range(len(base_vocab)), base_vocab))
        merges = []
        
        pbar = tqdm(total=self.vocab_size - len(vocab), desc="Merging byte pairs")
        
        for i in range(self.vocab_size - len(vocab)):
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                word_pairs = get_pairs(word)
                for pair in word_pairs:
                    pairs[pair] += freq
            
            if not pairs:
                break
            
            
            # # 关键：完全按照原始BPE算法选择
            # best = max(pairs, key=pairs.get)
            # merges.append(best)
            
            
            #! Debug: 找频率最高的pair
            max_freq = max(pairs.values())
            best_pairs = [pair for pair, freq in pairs.items() if freq == max_freq]
            
            # Tie-breaking: 选择字典序最大的pair，基于原始bytes
            # 根据CHANGELOG.md 0.1.1: "tiebreaking on this remapped unicode representation instead of the original bytes"
            # 我们应该基于原始bytes进行tie-breaking
            def get_bytes_key(pair):
                # 直接比较bytes对象
                return (pair[0], pair[1])
            
            # 选择原始bytes下字典序最大的pair
            best = max(best_pairs, key=get_bytes_key)
            
            
            
            vocab[len(vocab)] = best[0] + best[1]
            
            # 合并词汇
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == best[0] and word[i + 1] == best[1]:
                        new_word.append(best[0] + best[1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word = tuple(new_word)
                new_word_freqs[new_word] = freq
            
            word_freqs = new_word_freqs
            pbar.update(1)
        
        pbar.close()
        
        self.vocab = vocab
        self.merge_rules = merges
        
        return vocab, merges
    
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


if __name__ == "__main__":
    trainer = ByteBPETrainer(
        vocab_size=500,
        special_tokens=["<|endoftext|>"]
    )
    
    test_file_path = "tests/fixtures/corpus.en"
    vocab_dict, merged_rules = trainer.train(test_file_path, num_processes=8)
    
    trainer.save_vocab('vocab.json')
    trainer.save_merge_rules('merged_rules.txt')
    
    print(f"训练完成！词汇表大小: {len(vocab_dict)}")
    print(f"合并规则数量: {len(merged_rules)}")