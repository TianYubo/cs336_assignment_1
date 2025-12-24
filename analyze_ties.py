#!/usr/bin/env python3
# 分析所有的tie情况

import sys
sys.path.append('/Users/kyleee/code/cs336/assignment1-basics')

from cs336_basics.byte_based_bpe_corrected import ByteBPETrainer
from collections import Counter, defaultdict
from tests.common import gpt2_bytes_to_unicode

def analyze_all_ties():
    print('=== 分析所有tie情况 ===')
    
    # 重新训练
    trainer = ByteBPETrainer(vocab_size=500, special_tokens=["<|endoftext|>"])
    
    # 读取文件并处理
    with open('tests/fixtures/corpus.en', 'rb') as f:
        content = f.read()
    
    # 分词处理（和训练器中一样）
    import regex as re
    special_tokens_bytes = [b'<|endoftext|>']
    PAT_BYTES = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    delimiter_pattern = b"|".join(re.escape(token) for token in special_tokens_bytes)
    if delimiter_pattern:
        splited_chunks = re.split(delimiter_pattern, content)
    else:
        splited_chunks = [content]
    
    words = []
    for single_chunk in splited_chunks:
        if not single_chunk:
            continue
        for match in PAT_BYTES.finditer(single_chunk):
            words.append(match.group())
    
    word_freqs = Counter(words)
    
    # 初始化word_splits
    word_splits = {}
    for word in word_freqs:
        word_splits[word] = [bytes([b]) for b in word]
    
    def get_pairs_from_splits(word_splits, word_freqs):
        pairs = defaultdict(int)
        for word, splits in word_splits.items():
            freq = word_freqs[word]
            for i in range(len(splits) - 1):
                pair = (splits[i], splits[i+1])
                pairs[pair] += freq
        return pairs
    
    def merge_pair_in_splits(word_splits, pair):
        new_word_splits = {}
        for word, splits in word_splits.items():
            new_splits = []
            i = 0
            while i < len(splits):
                if i < len(splits) - 1 and splits[i] == pair[0] and splits[i+1] == pair[1]:
                    merged = splits[i] + splits[i+1]
                    new_splits.append(merged)
                    i += 2
                else:
                    new_splits.append(splits[i])
                    i += 1
            new_word_splits[word] = new_splits
        return new_word_splits
    
    # 读取参考merges
    with open('tests/fixtures/train-bpe-reference-merges.txt') as f:
        ref_lines = f.readlines()

    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    reference_merges = []
    for line in ref_lines:
        parts = line.strip().split(' ')
        if len(parts) == 2:
            token1, token2 = parts
            byte1 = bytes([gpt2_byte_decoder[token] for token in token1])
            byte2 = bytes([gpt2_byte_decoder[token] for token in token2])
            reference_merges.append((byte1, byte2))
    
    # 执行merge并分析ties
    merges = []
    tie_steps = []
    
    for step in range(len(reference_merges)):
        pairs = get_pairs_from_splits(word_splits, word_freqs)
        
        if not pairs:
            break
        
        # 找频率最高的pair
        max_freq = max(pairs.values())
        best_pairs = [pair for pair, freq in pairs.items() if freq == max_freq]
        
        if len(best_pairs) > 1:
            tie_steps.append(step)
            print(f'\\nStep {step} (Tie! {len(best_pairs)} options):')
            
            # 参考选择的
            ref_merge = reference_merges[step]
            ref_bytes = ref_merge[0] + ref_merge[1]
            print(f'  参考选择: {repr(ref_bytes)}')
            
            # 所有候选
            print(f'  所有候选:')
            for i, pair in enumerate(best_pairs):
                merged = pair[0] + pair[1]
                is_ref = merged == ref_bytes
                marker = "← 参考" if is_ref else ""
                len_info = f"len({len(pair[0])}+{len(pair[1])}={len(pair[0])+len(pair[1])})"
                print(f'    {i+1}: {repr(merged)} {len_info} {marker}')
        
        # 选择参考的选择（为了保持一致）
        ref_merge = reference_merges[step]
        best_pair = ref_merge
        
        merges.append(best_pair)
        
        # 执行merge
        word_splits = merge_pair_in_splits(word_splits, best_pair)
    
    print(f'\\n总共发现 {len(tie_steps)} 个tie步骤: {tie_steps}')

if __name__ == '__main__':
    analyze_all_ties()