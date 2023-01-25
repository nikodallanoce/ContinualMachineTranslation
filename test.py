import random
from typing import List

import numpy as np
import torch
from torch import Tensor
from transformers import MBartTokenizer


def permute_sentence(original_row: Tensor, start_of_sentence_indexes: list[int]):
    target_row = torch.ones_like(original_row)

    i = 0
    for start_idx in start_of_sentence_indexes:
        curr_token = int(original_row[start_idx])
        while curr_token not in special_ids and i < target_row.shape[0]:
            target_row[i] = curr_token
            i = i + 1
            start_idx = start_idx + 1
            curr_token = int(original_row[start_idx])
            if curr_token in end_of_sentence_ids:
                target_row[i] = curr_token
                i = i + 1
    if i < target_row.shape[0]:
        curr_token = int(original_row[i])
        while curr_token != tokenizer.pad_token_id and i < target_row.shape[0]:
            curr_token = int(original_row[i])
            target_row[i] = curr_token
            i = i + 1
    return target_row


def get_start_of_sentence_indexes(row_ids: Tensor):
    end_of_sentence_ids = {5, 32, 38}
    start_of_sentences: List[int] = [0]
    ends_with_punctuation = False
    for i in range(len(row_ids)):
        tok_ids = int(row_ids[i])
        if tok_ids == tokenizer.eos_token_id:
            if int(row_ids[i - 1]) in end_of_sentence_ids:
                ends_with_punctuation = True
            break
        if tok_ids in end_of_sentence_ids:
            start_of_sentences.append(i + 1)

    if ends_with_punctuation:
        start_of_sentences = start_of_sentences[: -1]
    return start_of_sentences


def mask_tokens(tokenized_ids: Tensor, seed: int, mask_percentage: float = 0.35):
    # replace some token that differ from the special ones with m_end mask token
    masked_ids = torch.ones_like(tokenized_ids)
    tokens_length = int((tokenized_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0])
    num_tokens_to_mask = round(tokens_length * mask_percentage)
    np.random.seed(seed)
    m_start = 0
    t_end = 0

    while num_tokens_to_mask > 0:
        mask_span_len = np.random.poisson(3.5)
        if mask_span_len > num_tokens_to_mask:
            mask_span_len = num_tokens_to_mask
        index = np.random.randint(t_end, tokens_length - num_tokens_to_mask)
        m_end = m_start + index - t_end
        masked_ids[m_start: m_end] = tokenized_ids[t_end: index]
        t_end = index + mask_span_len
        m_start = m_end
        num_tokens_to_mask = num_tokens_to_mask - mask_span_len
        if m_start > 0 and masked_ids[m_start - 1] != tokenizer.mask_token_id:
            masked_ids[m_start] = tokenizer.mask_token_id
            m_start = m_start + 1

    m_end = m_start + (tokens_length + 2 - t_end)
    masked_ids[m_start: m_end] = tokenized_ids[t_end: tokens_length + 2]
    return masked_ids


def internal_apply_noise(prefix, tokens):
    # print(prefix, tokens, file=sys.stderr)
    noise_mask_token = tokenizer.mask_token
    noise_token_replacement_rate = 0.35
    noise_span_avg_length = 3.5
    num_tokens_to_replace = int(len(tokens) * noise_token_replacement_rate)
    rv_tokens = []
    last_i = 0
    just_added_empty_span = False
    while (num_tokens_to_replace > 0):
        span_length = np.random.poisson(noise_span_avg_length)
        if span_length > num_tokens_to_replace:
            span_length = num_tokens_to_replace
        i = np.random.randint(last_i, len(tokens) - num_tokens_to_replace)
        rv_tokens.extend(tokens[last_i:i])
        if (len(rv_tokens) == 0) or (rv_tokens[-1] != noise_mask_token):
            rv_tokens.append(noise_mask_token)
        last_i = i + span_length
        num_tokens_to_replace -= span_length
    rv_tokens.extend(tokens[last_i:])
    return ' '.join(prefix + rv_tokens)


if __name__ == '__main__':
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")
    end_of_sentence_ids = {5, 32, 38}
    special_ids = set(tokenizer.all_special_ids).union(end_of_sentence_ids)

    # what = internal_apply_noise(["abc"], "Hello! How are you? Fine thanks. Hello! How are you? Fine thanks.".split(" "))

    original_row = tokenizer(
        'While her main work was in nutrition and physiology, she was also concerned with textiles, detergents, and dyes. She was technical advisor to the Pennsylvania Laundry Owners Association, and helped to develop the standards code of the Pennsylvania Association of Cleaners and Dyers.',
        max_length=128, truncation=True, padding='max_length',
        return_tensors='pt')
    a = original_row.input_ids.shape[1]
    original_row = original_row.input_ids.view(a)
    indexes = get_start_of_sentence_indexes(original_row)
    random.shuffle(indexes)
    perm_sent = permute_sentence(original_row, indexes)
    perm_sent = mask_tokens(perm_sent, 0)
