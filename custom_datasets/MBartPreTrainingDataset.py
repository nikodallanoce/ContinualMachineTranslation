from typing import List, Set, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch import Tensor
from transformers import MBartTokenizer
import datasets
from noise_functions.MBartNoiseFunction import MBartNoiseFunction
from utilities.utility import prepare_ds_indexes, retrieve_lang


class MBartPreTrainingDataset(Dataset):

    def __init__(self, hugg_datasets: Dict[str, datasets.Dataset], tokenizers: Dict[str, MBartTokenizer],
                 input_max_length=128):
        super(MBartPreTrainingDataset, self).__init__()
        self.ds_indexes: Dict[str, Tuple[int, int]] = prepare_ds_indexes(hugg_datasets)
        self.dataset: datasets.Dataset = datasets.concatenate_datasets(list(hugg_datasets.values()))
        self.tokenizers: Dict[str, MBartTokenizer] = tokenizers
        self.input_max_length = input_max_length
        self.noise_fn = MBartNoiseFunction()
        self.ref_len = input_max_length - input_max_length * 0.4

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        field = "text"
        label_ids = self.dataset[index][field]
        label_ids = label_ids.strip()
        lang, i_s, i_e = retrieve_lang(self.ds_indexes, index)
        tokenizer = self.tokenizers[lang]

        rng = np.random.default_rng(index)
        while len(label_ids.split(" ")) < self.ref_len:
            new_index = rng.integers(i_s, i_e, dtype=int)
            new_sent = self.dataset[new_index][field]
            if len(new_sent) < 4:
                continue
            label_ids += tokenizer.eos_token + " " + new_sent.strip()

        label_ids, masked_ids = self.noise_fn.compute(label_ids, index)
        tokenized = tokenizer([label_ids, masked_ids], return_special_tokens_mask=False,
                              add_special_tokens=True, truncation=True, return_attention_mask=False,
                              max_length=self.input_max_length, padding='longest',
                              return_tensors='pt')['input_ids']

        label_ids = tokenized[0].view(-1)
        masked_ids = tokenized[1].view(-1)

        att_mask = torch.where(masked_ids != 1, 1, 0)
        label_ids = torch.where(label_ids == 1, -100, label_ids)
        return {'input_ids': masked_ids, 'labels': label_ids, 'attention_mask': att_mask}

# def permute(self, tokenized_ids: Tensor, seed: int) -> Tensor:
#     sosi = self.get_punctuation_indexes(tokenized_ids)
#     permuted_sentences = self.permute_sentence(tokenized_ids, sosi, seed)
#     return permuted_sentences
#
# # def permute_sentence(self, original_row: Tensor, start_of_sentence_indexes: List[int]) -> Tensor:
# #     target_row = torch.ones_like(original_row)
# #     i = 0
# #     for start_idx in start_of_sentence_indexes:
# #         curr_token = int(original_row[start_idx])
# #         while curr_token not in self.special_ids_plus_eose and i < target_row.shape[0]:
# #             target_row[i] = curr_token
# #             i = i + 1
# #             start_idx = start_idx + 1
# #             curr_token = int(original_row[start_idx])
# #             if curr_token in self.end_of_sentence_ids:
# #                 target_row[i] = curr_token
# #                 i = i + 1
# #     if i < target_row.shape[0]:
# #         curr_token = int(original_row[i])
# #         while curr_token != self.tokenizer.pad_token_id and i < target_row.shape[0]:
# #             curr_token = int(original_row[i])
# #             target_row[i] = curr_token
# #             i = i + 1
# #     return target_row
#
# def permute_sentence(self, original_row: Tensor, start_of_sentence_indexes: List[int], seed: int) -> Tensor:
#     target_row = torch.ones_like(original_row)
#     i_max = start_of_sentence_indexes[-1]
#     rng = np.random.default_rng(seed)
#     # if i_max > self.imax:
#     #    self.imax = i_max
#     #    print(i_max)
#     start_end_sent_idx: List[Tuple[int, int]] = []
#     for i in range(len(start_of_sentence_indexes) - 1):
#         start_end_sent_idx.append((start_of_sentence_indexes[i], start_of_sentence_indexes[i + 1]))
#     rng.shuffle(start_end_sent_idx)
#
#     i = 0
#     for i_s, i_e in start_end_sent_idx:
#         sent_len = i_e - i_s
#         target_row[i:i + sent_len] = original_row[i_s:i_e]
#         i = i + sent_len
#     target_row[i:i + 2] = original_row[i_max:i_max + 2]
#     return target_row
#
# def get_punctuation_indexes(self, row_ids: Tensor) -> List[int]:
#     start_of_sentences: List[int] = [0]
#     # ends_with_punctuation = False
#     for i in range(len(row_ids)):
#         tok_ids = int(row_ids[i])
#         if tok_ids == self.tokenizer.eos_token_id:
#             if int(row_ids[i - 1]) not in self.end_of_sentence_ids:
#                 start_of_sentences.append(i)
#             break
#         if tok_ids in self.end_of_sentence_ids:
#             start_of_sentences.append(i + 1)
#
#     return start_of_sentences
#
# def mask_tokens(self, tokenized_ids: Tensor, seed: int, mask_percentage: float = 0.35):
#     # replace some token that differ from the special ones with a mask token
#     masked_ids = torch.ones_like(tokenized_ids)
#     tokens_length = int((tokenized_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0])
#     num_tokens_to_mask = round(tokens_length * mask_percentage)
#     rng = np.random.default_rng(seed)
#     m_start = 0
#     t_end = 0
#
#     while num_tokens_to_mask > 0:
#         mask_span_len = np.random.poisson(3.5)
#         if mask_span_len > num_tokens_to_mask:
#             mask_span_len = num_tokens_to_mask
#         index = rng.integers(t_end, tokens_length - num_tokens_to_mask)
#         m_end = m_start + index - t_end
#         masked_ids[m_start: m_end] = tokenized_ids[t_end: index]
#         t_end = index + mask_span_len
#         m_start = m_end
#         num_tokens_to_mask = num_tokens_to_mask - mask_span_len
#         if m_start > 0 and masked_ids[m_start - 1] != self.tokenizer.mask_token_id:
#             masked_ids[m_start] = self.tokenizer.mask_token_id
#             m_start = m_start + 1
#
#     m_end = m_start + (tokens_length + 2 - t_end)
#     masked_ids[m_start: m_end] = tokenized_ids[t_end: tokens_length + 2]
#     return masked_ids
