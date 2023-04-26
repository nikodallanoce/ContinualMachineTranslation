import copy
import time
from typing import List, Set, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch import Tensor
from transformers import MBartTokenizerFast
import datasets
from noise_functions.MBartNoiseFunction import MBartNoiseFunction
from utilities.utility import prepare_ds_indexes, retrieve_lang


class MBartPreTrainingDataset(Dataset):

    def __init__(self, hugg_dataset: datasets.Dataset, tokenizer: MBartTokenizerFast,
                 input_max_length=128, ds_field: str = "text"):
        super(MBartPreTrainingDataset, self).__init__()
        # self.ds_indexes: Dict[str, Tuple[int, int]] = prepare_ds_indexes(hugg_datasets)
        self.dataset: datasets.Dataset = hugg_dataset  # datasets.concatenate_datasets(list(hugg_datasets.values()))
        self.tokenizer: MBartTokenizerFast = tokenizer
        self.input_max_length: int = input_max_length
        self.noise_fn = MBartNoiseFunction()
        self.ds_field: str = ds_field
        self.ref_len: float = input_max_length - input_max_length * 0.4

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        field = self.ds_field
        sampled_sent_min_len = 4
        label_ids = self.dataset[index][field]

        rng = np.random.default_rng(index)
        while len(label_ids.split(" ")) < sampled_sent_min_len:
            new_index = rng.integers(0, len(self.dataset) - 1, dtype=int)
            label_ids = self.dataset[new_index][field]

        label_ids = label_ids.strip()
        while len(label_ids.split(" ")) < self.ref_len:
            new_index = rng.integers(0, len(self.dataset) - 1, dtype=int)
            new_sent = self.dataset[new_index][field]
            if len(new_sent) < sampled_sent_min_len:
                continue
            label_ids += self.tokenizer.eos_token + " " + new_sent.strip()

        label_ids, masked_ids = self.noise_fn.compute(label_ids, index)
        tokenized = self.tokenizer([label_ids, masked_ids], return_special_tokens_mask=False,
                                   add_special_tokens=True, truncation=True, return_attention_mask=False,
                                   max_length=self.input_max_length, padding='longest',
                                   return_tensors='pt')['input_ids']

        label_ids = tokenized[0].view(-1)
        masked_ids = tokenized[1].view(-1)

        att_mask = torch.where(masked_ids != self.tokenizer.pad_token_id, 1, 0)
        label_ids = torch.where(label_ids == self.tokenizer.pad_token_id, -100, label_ids)
        return {'input_ids': masked_ids, 'labels': label_ids, 'attention_mask': att_mask}


# def get_item_for_iterable(batch, tokenizer: MBartTokenizerFast, noise_fn: MBartNoiseFunction = MBartNoiseFunction(),
#                           seed: int = 0,
#                           input_max_length: int = 128,
#                           num_threads: int = 4):
#     if type(batch) is list:
#         from concurrent.futures import ThreadPoolExecutor
#         labels_lst = []
#         masked_ids_lst = []
#         att_mask_lst = []
#         batch_per_thr, batch_reminder = divmod(len(batch), num_threads)
#         i_start = 0
#         i_end = 0
#         results = []
#         with ThreadPoolExecutor(num_threads) as executor:
#             for i in range(num_threads):
#                 batch_per_thr = batch_per_thr + int(batch_reminder > 0)
#                 batch_reminder -= 1
#                 i_end += batch_per_thr
#                 txt_batch = batch[i_start: i_end]
#                 results.append(executor.submit(
#                     thr_batch_tokenize, batch=txt_batch, input_max_length=input_max_length, noise_fn=noise_fn,
#                     seed=seed,
#                     tokenizer=copy.copy(tokenizer)))
#                 i_start = i_end
#     else:
#         att_mask_lst, label_ids_lst, masked_ids_lst = noise_and_tokenize(input_max_length, noise_fn, seed, batch,
#                                                                          tokenizer)
#     return {'input_ids': masked_ids_lst, 'labels': labels_lst, 'attention_mask': att_mask_lst}

def get_item_for_iterable(batch, tokenizer: MBartTokenizerFast,
                          noise_fn: MBartNoiseFunction = MBartNoiseFunction(),
                          seed: int = 0,
                          input_max_length: int = 128):
    if type(batch) is list:
        labels_lst = []
        masked_ids_lst = []
        att_mask_lst = []

        for text in batch:
            att_mask, label, masked_ids = noise_and_tokenize(input_max_length, noise_fn, seed, text, tokenizer)
            labels_lst.append(label)
            att_mask_lst.append(att_mask)
            masked_ids_lst.append(masked_ids)

    else:
        att_mask_lst, labels_lst, masked_ids_lst = noise_and_tokenize(input_max_length, noise_fn, seed, batch,
                                                                      tokenizer)
    return {'input_ids': masked_ids_lst, 'labels': labels_lst, 'attention_mask': att_mask_lst}


def thr_batch_tokenize(batch, input_max_length: int, noise_fn: MBartNoiseFunction, seed: int,
                       tokenizer: MBartTokenizerFast):
    labels_lst = []
    masked_ids_lst = []
    att_mask_lst = []
    for text in batch:
        att_mask, label_ids, masked_ids = noise_and_tokenize(input_max_length, noise_fn, seed, text, tokenizer)
        masked_ids_lst.append(masked_ids)
        att_mask_lst.append(att_mask)
        labels_lst.append(label_ids)
    return att_mask_lst, labels_lst, masked_ids_lst


def noise_and_tokenize(input_max_length, noise_fn, seed, text, tokenizer):
    text = text.strip()
    ref_len = round(input_max_length - 0.4 * input_max_length)
    text = " ".join(text.split()[0:ref_len])
    label_ids, masked_ids = noise_fn.compute(text, seed)
    tokenized = tokenizer([label_ids, masked_ids], return_special_tokens_mask=False,
                          add_special_tokens=True, truncation=True, return_attention_mask=False,
                          max_length=input_max_length, padding='longest')['input_ids']
    label_ids = tokenized[0]
    masked_ids = tokenized[1]
    att_mask = [1 if e != tokenizer.pad_token_id else 0 for e in masked_ids]
    label_ids = [-100 if e == tokenizer.pad_token_id else e for e in label_ids]
    return att_mask, label_ids, masked_ids
