import os
from typing import BinaryIO
import regex as re
from collections import Counter
from itertools import chain
from tqdm import tqdm
from typing import Tuple, Set, Dict, List
import multiprocessing
import json
from functools import lru_cache

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_BYTES = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")



class BPETrainer:
    """Byte Pair Encoding Trainer"""
    
    def __init__(self, vocab_size: int, special_tokens: List[str] = None):
        """
        初始化BPE训练器
        
        Args:
            vocab_size: 目标词汇表大小
            special_tokens: 特殊token列表，默认包含 <|endoftext|>
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<|endoftext|>"]
        self.vocab = {}
        self.merge_rules = []
        
        # 预分词正则表达式
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    
    @staticmethod
    @lru_cache
    def gpt2_bytes_to_unicode() -> Dict[int, str]:
        """GPT-2字节到Unicode字符的映射"""
        # 原始实现保持不变
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        characters = [chr(n) for n in cs]
        d = dict(zip(bs, characters))
        return d
    
    
    def build_unicode_to_byte_map(self) -> Dict[str, int]:
        """构建反向映射：字符 → 原始字节"""
        byte_to_char = self.gpt2_bytes_to_unicode()
        return {char: byte for byte, char in byte_to_char.items()}
    
    
    def get_reference_vocab_bytes(self) -> List[bytes]:
        """
        根据你提供的“字符列表”，返回每个字符对应的真实原始字节（GPT-2 编码方式）
        输出是 256 个 bytes 对象，如 b'\x00', b'\x01', ..., b'\xff'
        """
        base_chars = [
            # ASCII可打印字符 (索引1-94) → 94 个
            "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "[", "\\", "]", "^", "_", "`",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            "{", "|", "}", "~",  # ← 这是第94个

            # 扩展ASCII字符 (索引95-188) → 应该是 94 个？
            "¡", "¢", "£", "¤", "¥", "¦", "§", "¨", "©", "ª", "«", "¬", "®", "¯", "°", "±", "²", "³", "´", "µ", "¶", "·", "¸", "¹", "º", "»", "¼", "½", "¾", "¿",  # 30
            "À", "Á", "Â", "Ã", "Ä", "Å", "Æ", "Ç", "È", "É", "Ê", "Ë", "Ì", "Í", "Î", "Ï", "Ð", "Ñ", "Ò", "Ó", "Ô", "Õ", "Ö", "×", "Ø", "Ù", "Ú", "Û", "Ü", "Ý", "Þ", "ß",  # 32 → 总62
            "à", "á", "â", "ã", "ä", "å", "æ", "ç", "è", "é", "ê", "ë", "ì", "í", "î", "ï", "ð", "ñ", "ò", "ó", "ô", "õ", "ö", "÷", "ø", "ù", "ú", "û", "ü", "ý", "þ", "ÿ",  # 32 → 总94

            # 拉丁扩展字符 (索引189-255) → 255 - 188 = 67 个
            "Ā", "ā", "Ă", "ă", "Ą", "ą", "Ć", "ć", "Ĉ", "ĉ", "Ċ", "ċ", "Č", "č", "Ď", "ď", "Đ", "đ", "Ē", "ē", "Ĕ", "ĕ", "Ė", "ė", "Ę", "ę", "Ě", "ě",  # 28
            "Ĝ", "ĝ", "Ğ", "ğ", "Ġ", "ġ", "Ģ", "ģ", "Ĥ", "ĥ", "Ħ", "ħ", "Ĩ", "ĩ", "Ī", "ī", "Ĭ", "ĭ", "Į", "į", "İ", "ı", "Ĳ", "ĳ", "Ĵ", "ĵ",  # 26 → 总54
            "Ķ", "ķ", "ĸ", "Ĺ", "ĺ", "Ļ", "ļ", "Ľ", "ľ", "Ŀ", "ŀ", "Ł", "ł", "Ń",  # 14 → 总68 ❗你写的是到255，但这里只有68个，不是67
        ]
        
        assert len(base_chars) == 256, f"字符列表长度必须是256，当前是{len(base_chars)}"

        # 获取反向映射：字符 → 原始字节
        char_to_byte = self.build_unicode_to_byte_map()

        bytes_list = []
        for i, char in enumerate(base_chars):
            if char not in char_to_byte:
                raise ValueError(f"字符 {repr(char)} 不在 GPT-2 映射表中，位置 {i}")
            original_byte = char_to_byte[char]
            bytes_list.append(bytes([original_byte]))

        return bytes_list
    
    
    def find_chunk_boundaries(self, file: BinaryIO, desired_num_chunks: int, 
                              split_special_token: bytes) -> List[int]:
        """将文件分块处理的边界查找"""
        # 原始实现保持不变
        assert isinstance(split_special_token, bytes), (
            "Must represent special token as a bytestring"
        )

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    
    def merge_all_pairs(self, char_tuple: tuple, target_pair: tuple):
        """合并元组中所有出现的目标字符对"""
        char_list = list(char_tuple)
        merged_token = ''.join(target_pair)
        
        # 从后往前处理，避免索引变化的问题
        i = len(char_list) - 2
        while i >= 0:
            if (char_list[i], char_list[i+1]) == target_pair:
                char_list[i] = merged_token
                char_list.pop(i+1)
            i -= 1
            
        return tuple(char_list)
    
    
    def get_word_frequency_in_chunk(self, chunk: str) -> Dict[tuple, int]:
        """获取文本块中的单词频率"""
        delimiter = "|".join(re.escape(token) for token in self.special_tokens)
        splited_chunks = re.split(delimiter, chunk)
        
        words = []
        for single_chunk in splited_chunks:
            for match in re.finditer(self.PAT, single_chunk):
                words.append(match.group())
                
        word_freq_counter = Counter(words)
        word_char_tuples = {
            tuple(list(word)): freq
            for word, freq in word_freq_counter.items()
        }
        
        return word_char_tuples
    
    
    def get_pairs(self, word_tuple: tuple) -> zip:
        """从单词的字符元组中提取所有相邻对"""
        return zip(word_tuple, word_tuple[1:])
    
    
    def bpe_merge_efficient_optimized(self, word_char_tuples: Dict[tuple, int], 
                                    vocab: Dict[int, bytes]) -> Tuple[Dict[int, bytes], List[tuple]]:
        """高效的BPE合并算法"""
        merged_rules = []
        
        # 初始计算所有字节对的频率
        pair_freqs = Counter()
        for word_tuple, freq in word_char_tuples.items():
            for pair in self.get_pairs(word_tuple):
                pair_freqs[pair] += freq
        
        pbar = tqdm(total=self.vocab_size - len(vocab), desc="Merging tokens")
        num_merges = self.vocab_size - len(vocab)
        
        for i in range(num_merges):
            if not pair_freqs:
                break
            
            # 查找最高频对
            max_freq = max(pair_freqs.values())
            candidates = [p for p, f in pair_freqs.items() if f == max_freq]
            best_pair = max(candidates)  # 字典序最小
            
            # 创建新token并加入词汇表
            new_token = "".join(best_pair)
            token_id = len(vocab)
            vocab[token_id] = new_token.encode('utf-8')
            
            # 记录合并规则
            best_pair_in_bytes = (best_pair[0].encode('utf-8'), 
                                best_pair[1].encode('utf-8'))
            merged_rules.append(best_pair_in_bytes)
            
            # 更新词频和字节对频率
            next_word_char_tuples = {}
            for word_tuple, freq in word_char_tuples.items():
                has_pair = any((word_tuple[j], word_tuple[j+1]) == best_pair 
                            for j in range(len(word_tuple) - 1))
                
                if not has_pair:
                    next_word_char_tuples[word_tuple] = next_word_char_tuples.get(word_tuple, 0) + freq
                    continue
                
                # 减去旧字节对频率
                for pair in self.get_pairs(word_tuple):
                    pair_freqs[pair] -= freq
                
                # 创建新单词元组
                j = 0
                new_word_list = []
                while j < len(word_tuple):
                    if j < len(word_tuple) - 1 and (word_tuple[j], word_tuple[j+1]) == best_pair:
                        new_word_list.append(new_token)
                        j += 2
                    else:
                        new_word_list.append(word_tuple[j])
                        j += 1
                new_word_tuple = tuple(new_word_list)
                
                next_word_char_tuples[new_word_tuple] = next_word_char_tuples.get(new_word_tuple, 0) + freq
                
                # 加上新字节对频率
                for pair in self.get_pairs(new_word_tuple):
                    pair_freqs[pair] += freq
            
            word_char_tuples = next_word_char_tuples
            pair_freqs = Counter({p: f for p, f in pair_freqs.items() if f > 0})
            
            pbar.update(1)
        
        pbar.close()
        return vocab, merged_rules
    
    
    def process_chunk(self, args: Tuple[int, int, str]) -> Tuple[Dict[tuple, int], Set[str]]:
        """处理单个文本块"""
        start, end, input_path = args
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # chunk = f.read(end - start)
        
        chunk_all_chars = set(chunk)
        chunk_word_char_tuples = self.get_word_frequency_in_chunk(chunk)
        
        return chunk_word_char_tuples, chunk_all_chars
    
    
    def train(self, input_path: str, num_processes: int = 8) -> Tuple[Dict[int, bytes], List[tuple]]:
        """
        训练BPE模型
        
        Args:
            input_path: 训练文本文件路径
            num_processes: 并行处理进程数
            
        Returns:
            vocab: 词汇表字典 {token_id: token_bytes}
            merge_rules: 合并规则列表
        """
        # 初始化基础词汇表
        base_vocab = []
        
        # 添加special tokens
        for token in self.special_tokens:
            base_vocab.append(token.encode('utf-8'))
        
        # 添加基础字符
        base_chars = self.get_reference_vocab_bytes()
        base_vocab.extend(base_chars)
        
        # 并行处理文本块
        with open(input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, num_processes, 
                                                    "<|endoftext|>".encode("utf-8"))
            
            word_char_tuples = Counter()
            all_chars_text = set()
            
            with multiprocessing.Pool(processes=num_processes) as pool:
                tasks = [(start, end, input_path) 
                        for start, end in zip(boundaries[:-1], boundaries[1:])]
                results = pool.map(self.process_chunk, tasks)
            
            # 合并结果
            for chunk_word_char_tuples, chunk_all_chars in results:
                word_char_tuples += chunk_word_char_tuples
                all_chars_text.update(chunk_all_chars)
        
        # 构建初始词汇表
        base_vocab_dict = dict(zip(range(len(base_vocab)), base_vocab))
        
        # 执行BPE合并
        self.vocab, self.merge_rules = self.bpe_merge_efficient_optimized(
            word_char_tuples, base_vocab_dict)
        
        return self.vocab, self.merge_rules
    
    
    def save_vocab(self, vocab_path: str):
        """保存词汇表到JSON文件"""
        saved_vocab_dict = {}
        for token_id, token in self.vocab.items():
            char_token = token.decode('utf-8')
            char_token = char_token.replace(' ', 'Ġ')
            saved_vocab_dict[char_token] = token_id
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(saved_vocab_dict, f, ensure_ascii=False, indent=4)
    
    
    def save_merge_rules(self, rules_path: str):
        """保存合并规则到文本文件"""
        with open(rules_path, 'w', encoding='utf-8') as f:
            for item1, item2 in self.merge_rules:
                str1 = item1.decode('utf-8').replace(' ', 'Ġ')
                str2 = item2.decode('utf-8').replace(' ', 'Ġ')
                f.write(f"{str1} {str2}\n")


# 使用示例
if __name__ == "__main__":
    # 创建BPE训练器
    trainer = BPETrainer(
        vocab_size=500,
        special_tokens=["<|endoftext|>"]
    )
    
    # 训练模型
    test_file_path = "tests/fixtures/corpus.en"
    vocab_dict, merged_rules = trainer.train(test_file_path)
    
    # 保存结果
    trainer.save_vocab('vocab.json')
    trainer.save_merge_rules('merged_rules.txt')
    
    print(f"训练完成！词汇表大小: {len(vocab_dict)}")
    print(f"合并规则数量: {len(merged_rules)}")