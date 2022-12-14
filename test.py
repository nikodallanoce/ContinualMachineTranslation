import random
from typing import List

import torch
from torch import Tensor
from transformers import MBartTokenizer


def permute_sentence(original_row: Tensor, start_of_sentence_indexes: list[int]):
    end_of_sentence_ids = {5, 32, 38}
    special_ids = set(tokenizer.all_special_ids).union(end_of_sentence_ids)
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


if __name__ == '__main__':
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")
    original_row = tokenizer('Hello! How are you? Fine thanks.', max_length=10, truncation=True, padding='max_length',
                             return_tensors='pt')
    a = original_row.input_ids.shape[1]
    original_row = original_row.input_ids.view(a)
    indexes = get_start_of_sentence_indexes(original_row)
    random.shuffle(indexes)
    permute_sentence(original_row, indexes)
