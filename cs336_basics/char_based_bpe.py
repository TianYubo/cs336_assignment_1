import os
from typing import BinaryIO
import regex as re
from collections import Counter
from itertools import chain
from tqdm import tqdm
from typing import Tuple, Set
import multiprocessing
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge_all_pairs(char_tuple, target_pair):
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


def smart_read_file(
    file_path, 
    size_threshold=100*1024*1024,
    chunk_size=8192):

    file_size = os.path.getsize(file_path)
    if file_size < size_threshold:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
        return ''.join(chunks)
    
    
from functools import lru_cache

@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.
    """
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

def build_unicode_to_byte_map() -> dict[str, int]:
    """
    构建反向映射：字符 → 原始字节 (0-255)
    """
    byte_to_char = gpt2_bytes_to_unicode()
    return {char: byte for byte, char in byte_to_char.items()}

def get_reference_vocab_bytes() -> list[bytes]:
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
    char_to_byte = build_unicode_to_byte_map()

    bytes_list = []
    for i, char in enumerate(base_chars):
        if char not in char_to_byte:
            raise ValueError(f"字符 {repr(char)} 不在 GPT-2 映射表中，位置 {i}")
        original_byte = char_to_byte[char]
        bytes_list.append(bytes([original_byte]))

    return bytes_list



def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
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

#* 原始实现
def bpe_chunk_pretokenization(
    chunk: str,
    vocab_size: int,
    special_tokens: list[str]):
    
    delimiter = "|".join(re.escape(token) for token in special_tokens)
    splited_chunks = re.split(delimiter, chunk)
    
    # 统计单词词频
    words = []
    for single_chunk in splited_chunks:
        for match in re.finditer(r'\w+', single_chunk.lower()):
            words.append(match.group())
    word_freq_counter = Counter(words)
    
    # 单词 : 频率 转换为 单词当中每一个字符 : 频率
    char_freq_counter = {}
    for word, freq in word_freq_counter.items():
        char_tuple = tuple(list(word) + ['</w>'])
        char_freq_counter[char_tuple] = freq
        
    # 初始化词表
    all_chars_dict = set(chain.from_iterable(char_freq_counter.keys()))
    # 单独把所有 special tokens 加入 all_chars_dict
    for tk in special_tokens:
        all_chars_dict.add(tk)

    #* 统计相邻符号对的频率
    all_merge_rules = dict()
    
    pbar = tqdm(desc="Processing", unit="iter")
    merge_counter = 0
    
    while len(all_chars_dict) < vocab_size:
        closest_pairs = Counter()
        #TODO: 目前是遍历所有单词的所有相邻对，然后统计最大值
        for cur_str, word_freq in char_freq_counter.items():
            pairs = [cur_str[i:i+2] for i in range(len(cur_str)-1)]
            pair_counts = {pair: word_freq for pair in pairs}
            closest_pairs.update(pair_counts)
        if not closest_pairs:
            break  # 没有更多可合并的对
        most_frequent_pair, frequency = closest_pairs.most_common(1)[0]
        
        # 合并最高频的字符对
        all_chars_dict.add(''.join(most_frequent_pair))
        all_merge_rules[most_frequent_pair] = ''.join(most_frequent_pair)
        
        new_char_freq = Counter()
        for cur_str, word_freq in char_freq_counter.items():
            merged_tuple = merge_all_pairs(cur_str, most_frequent_pair)
            new_char_freq[merged_tuple] = word_freq
        char_freq_counter = new_char_freq
        
        #* 只是作为一个进度条
        merge_counter += 1
        pbar.update(1)
        pbar.set_postfix({"Current": merge_counter})
    
    return all_chars_dict, all_merge_rules


# def getWordFrequencyInChunk(chunk: str, special_tokens: list[str]):
#     delimiter = "|".join(re.escape(token) for token in special_tokens)
#     splited_chunks = re.split(delimiter, chunk)
    
#     base_vocab = set()
    
        
#     # 统计单词词频，使用预分词正则化表示
#     words = []
#     for single_chunk in splited_chunks:
#         # base_vocab.update(single_chunk)
#         for match in re.finditer(PAT, single_chunk):
#             words.append(match.group())
            
#     word_freq_counter = Counter(words)
#     word_char_tuples = {
#         tuple(list(word)): freq
#         for word, freq in word_freq_counter.items()
#     }

#     # # 收集基础字符
#     # for tup in word_char_tuples.keys():
#     #     base_vocab.update(tup)

#     return word_char_tuples, base_vocab


def getWordFrequencyInChunk(chunk: str, special_tokens: list[str], reference_vocab_file: str = None):
    delimiter = "|".join(re.escape(token) for token in special_tokens)
    splited_chunks = re.split(delimiter, chunk)
    
    
        
    # 统计单词词频
    words = []
    for single_chunk in splited_chunks:
        for match in re.finditer(PAT, single_chunk):
            words.append(match.group())
            
    word_freq_counter = Counter(words)
    word_char_tuples = {
        tuple(list(word)): freq
        for word, freq in word_freq_counter.items()
    }

    return word_char_tuples



    
def get_pairs(word_tuple):
    """从单词的字符元组中提取所有相邻对"""
    return zip(word_tuple, word_tuple[1:])


def bpe_merge_efficient_optimized(
    word_char_tuples: dict[tuple, int],
    vocab: dict[int, bytes],
    vocab_size: int):
    """
    更高效的 BPE 训练实现 (优化版)
    :param word_char_tuples: 一个字典，键是 (字符,...) 元组，值是频率
    :param vocab: 初始词汇表集合
    :param vocab_size: 目标词汇表大小
    """
    merged_rules = []

    # 步骤 1: 初始计算所有字节对的频率
    pair_freqs = Counter()
    for word_tuple, freq in word_char_tuples.items():
        for pair in get_pairs(word_tuple):
            pair_freqs[pair] += freq
    pbar = tqdm(total=vocab_size - len(vocab), desc="Merging tokens")
    num_merges = vocab_size - len(vocab)
    
    for i in range(num_merges):
        if not pair_freqs:
            break

        # 步骤 2: 查找最高频对
        # 在 `max` 的 `key` 中，我们使用一个元组 `(freq, pair)` 来比较。
        # Python 会先比较元组的第一个元素（频率），如果频率相同，则比较第二个元素（字节对本身），
        # 这会自动处理平局情况，选择字典序更小的字节对。
        # 但由于我们只需要最高频，直接用 `max(pair_freqs.values())` 然后再查找更高效。
        # best_pair = max(pair_freqs, key=pair_freqs.get)
        # best_pair = max(pair_freqs, key=lambda k: (pair_freqs.get(k), k))
        
        max_freq = max(pair_freqs.values())
        candidates = [p for p,f in pair_freqs.items() if f == max_freq]
        best_pair = max(candidates)  # 字典序最小

        # 步骤 3: 创建新 token 并加入词汇表
        new_token = "".join(best_pair)
        # vocab.add(new_token.encode('utf-8'))
        token_id = len(vocab)
        vocab[token_id] = new_token.encode('utf-8')
        
        # 记录合并规则
        # 确保以 bytes 形式存储规则，与 huggingface/tokenizers 兼容
        best_pair_in_bytes = (best_pair[0].encode('utf-8'), 
                              best_pair[1].encode('utf-8'))
        merged_rules.append(best_pair_in_bytes)

        # ========================= 核心修正点 =========================
        # 采用更稳健的“先减后加”策略更新频率
        next_word_char_tuples = {}
        for word_tuple, freq in word_char_tuples.items():
            
            # 检查当前单词是否受影响，如果不受影响，直接跳过
            # 这比 `if best_pair in get_pairs(word)` 更高效，因为它只检查一次
            has_pair = any((word_tuple[j], word_tuple[j+1]) == best_pair for j in range(len(word_tuple) - 1))
            if not has_pair:
                # 不受影响的单词直接加入下一次迭代
                next_word_char_tuples[word_tuple] = next_word_char_tuples.get(word_tuple, 0) + freq
                continue

            # --- 对受影响的单词执行合并和频率更新 ---
            # 1. 减去这个单词中所有旧字节对的频率
            for pair in get_pairs(word_tuple):
                pair_freqs[pair] -= freq

            # 2. 创建新的单词元组
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
            
            # 3. 将新单词及其频率添加到下一次迭代的字典中（注意累加频率）
            next_word_char_tuples[new_word_tuple] = next_word_char_tuples.get(new_word_tuple, 0) + freq
            
            # 4. 加上新单词中所有新字节对的频率
            for pair in get_pairs(new_word_tuple):
                pair_freqs[pair] += freq

        # 使用新生成的单词字典进行下一次迭代
        word_char_tuples = next_word_char_tuples
        # 清理频率为 0 或负数的字节对，并移除已合并的 best_pair
        pair_freqs = Counter({p: f for p, f in pair_freqs.items() if f > 0})
        
        pbar.update(1)

    pbar.close()
    return vocab, merged_rules


# 定义处理单个 chunk 的函数
def process_chunk(args: Tuple[int, int, str, list]):
    start, end, input_path, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    chunk_all_chars = set()
    chunk_all_chars.update(chunk)
    
    chunk_word_char_tuples = getWordFrequencyInChunk(chunk, special_tokens)
    return chunk_word_char_tuples, chunk_all_chars


def bpe_trainer_main(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,):
    
    # 初始化基础词汇表
    base_vocab = []
    
    # 1. 首先添加special tokens
    for token in special_tokens:
        base_vocab.append(token.encode('utf-8'))
    # base_chars = get_base_character_set()
    base_chars = get_reference_vocab_bytes()
    base_vocab.extend(base_chars)
    
    # 统计词频以及出现的所有字符
    num_processes = 8
    with open(input_path, "rb") as f:
        
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
        actual_boundaries = boundaries
        word_char_tuples = Counter()
        
        # 并行处理
        with multiprocessing.Pool(processes=num_processes) as pool:
            # 准备任务列表
            tasks = [(start, end, input_path, special_tokens) 
                    for start, end in zip(actual_boundaries[:-1], actual_boundaries[1:])]
            # 并行执行
            results = pool.map(process_chunk, tasks)
        
        # 合并结果
        all_chars_text = set()
        for chunk_word_char_tuples, chunk_all_chars in results:
            word_char_tuples += chunk_word_char_tuples
            all_chars_text.update(chunk_all_chars)
            
            
    base_vocab_dict = dict(zip(range(0, len(base_vocab)), base_vocab))
    # base_vocab_dict = dict(zip(base_vocab, range(0, len(base_vocab))))
    
    # # 串行实现方法
    # num_processes = 256
    # with open(input_path, "rb") as f:
    #     boundaries = find_chunk_boundaries(
    #         f, num_processes, "<|endoftext|>".encode("utf-8"))
        
    #     actual_boundaries = boundaries
    #     word_char_tuples = Counter()
    #     vocab = set()
        
    #     # The following is a serial implementation, but you can parallelize this 
    #     # by sending each start/end pair to a set of processes.
    #     for start, end in zip(actual_boundaries[:-1], actual_boundaries[1:]):
    #         f.seek(start)
    #         chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #         chunk_word_char_tuples, chunk_vocab = getWordFrequencyInChunk(chunk, special_tokens)
    #         word_char_tuples += chunk_word_char_tuples
    #         vocab.update(chunk_vocab)
    
    #TODO: 有的单词前面没有空格，有的有
    bytes_vocab = {}
    bytes_vocab, merged_rules = bpe_merge_efficient_optimized(word_char_tuples, base_vocab_dict, vocab_size)
    
    # for idx, token in vocab.items():
    #     # 将空格替换为 Ġ，然后编码为UTF-8
    #     # processed_token = token.replace(' ', 'Ġ')
    #     processed_token = token
    #     bytes_vocab[idx] = processed_token.encode('utf-8')
    
    # vocab = dict(enumerate(vocab))
    return bytes_vocab, merged_rules


def process_vocab_item(item):
    """处理词汇表中的单个元素"""
    # 如果是 bytes 类型，转换为 str
    if isinstance(item, bytes):
        item = item.decode('utf-8')
    # 将空格替换为 Ġ
    item = item.replace(' ', 'Ġ')
    return item


if __name__ == "__main__":
    # test_file_path = "data/TinyStoriesV2-GPT4-valid.txt"
    test_file_path = "/Users/kyleee/code/cs336/assignment1-basics/tests/fixtures/corpus.en"
    vocab_dict, merged_rules = bpe_trainer_main(
        input_path=test_file_path,
        vocab_size=500,
        # special_tokens=["<|endoftext|>", "<|startoftext|>", "[SEP]"]
        special_tokens=["<|endoftext|>",]
    )
    
    # 保存到文件
    with open('merged_rules.txt', 'w', encoding='utf-8') as f:
        for item1, item2 in merged_rules:
            # 将 bytes 转换为字符串
            str1 = item1.decode('utf-8')
            str2 = item2.decode('utf-8')
            
            # 将空格替换为 Ġ 符号
            str1 = str1.replace(' ', 'Ġ')
            str2 = str2.replace(' ', 'Ġ')
            
            # 写入文件，每行一组，用空格分隔
            f.write(f"{str1} {str2}\n")
    
    saved_vocab_dict = dict()
    for token_id, token in vocab_dict.items():
        char_token = token.decode('utf-8')
        char_token = char_token.replace(' ', 'Ġ')
        saved_vocab_dict[char_token] = token_id
    
    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump(saved_vocab_dict, f, ensure_ascii=False, indent=4)
    
    