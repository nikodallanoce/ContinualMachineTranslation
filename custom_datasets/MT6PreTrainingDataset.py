import random
import sys
from typing import Dict, Tuple, List, Optional, Union

import datasets
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import MT5TokenizerFast

from noise_functions.MT5NoiseFunction import MT5NoiseFunction
from noise_functions.MT6NoiseFunction import MT6NoiseFunction


class MT6PreTrainingDataset(Dataset):

    def __init__(self, hugg_dataset: datasets.Dataset, tokenizer: MT5TokenizerFast, input_max_length=128,
                 ds_field: str = "text",
                 noise_fn: MT6NoiseFunction = MT6NoiseFunction(pnat=True)):
        super().__init__()
        # self.ds_indexes: Dict[str, Tuple[int, int]] = prepare_ds_indexes(hugg_datasets)
        self.dataset: datasets.Dataset = hugg_dataset  # datasets.concatenate_datasets(list(hugg_datasets.values()))
        self.tokenizer: MT5TokenizerFast = tokenizer
        # self.labels_name: str = labels_name
        self.input_max_length: int = input_max_length
        self.ds_field: str = ds_field
        self.noise_fn: Union[MT6NoiseFunction, MT5NoiseFunction] = noise_fn
        self.ref_len: int = round(input_max_length - input_max_length * 0.35)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        field = self.ds_field
        sampled_sent_min_len = 4
        text = self.dataset[index][field]

        rng = np.random.default_rng(index)
        while len(text.split(" ")) < sampled_sent_min_len:
            new_index = rng.integers(0, len(self.dataset) - 1, dtype=int)
            text = self.dataset[new_index][field]

        text = text.strip()
        while len(text.split(" ")) < self.ref_len:
            new_index = rng.integers(0, len(self.dataset) - 1, dtype=int)
            new_text = self.dataset[new_index][field]
            text = text + " " + new_text.strip()

        att_mask, labels, input_ids = noise_and_tokenize(self.input_max_length, self.noise_fn, index, text,
                                                         self.tokenizer, self.ref_len, tensor_version=True)

        outputs = {"input_ids": input_ids, "labels_pnat": labels, "attention_mask": att_mask}

        return outputs


def get_item_for_iterable(batch, tokenizer: MT5TokenizerFast,
                          noise_fn: MT6NoiseFunction = MT6NoiseFunction(pnat=True),
                          seed: int = None,
                          input_max_length: int = 128,
                          has_translation_pairs: bool = False):
    labels = "labels_pnat"
    if has_translation_pairs:
        # if not isinstance(noise_fn, MT5NoiseFunction):
        #     noise_fn = MT5NoiseFunction()
        labels = "labels_transl"

    if type(batch) is list:
        labels_lst = []
        input_ids_lst = []
        att_mask_lst = []

        for text in batch:

            seed = np.random.randint(0, 2147483648)
            if has_translation_pairs:
                att_mask, label, input_ids = translation_span_corruption(input_max_length, noise_fn, seed, text,
                                                                         tokenizer)

            else:
                att_mask, label, input_ids = noise_and_tokenize(input_max_length, noise_fn, seed, text, tokenizer)

            labels_lst.append(label)
            att_mask_lst.append(att_mask)
            input_ids_lst.append(input_ids)

    else:
        if has_translation_pairs:
            att_mask_lst, labels_lst, input_ids_lst = translation_span_corruption(input_max_length, noise_fn, seed,
                                                                                  batch, tokenizer)
        else:
            att_mask_lst, labels_lst, input_ids_lst = noise_and_tokenize(input_max_length, noise_fn, seed, batch,
                                                                         tokenizer)

    return {'input_ids': input_ids_lst, labels: labels_lst, 'attention_mask': att_mask_lst}


def translation_span_corruption(input_max_length: int, noise_fn: Union[MT5NoiseFunction, MT6NoiseFunction], seed: int,
                                text_pair: Dict[str, str],
                                tokenizer: MT5TokenizerFast):
    rng = np.random.default_rng(seed)
    transl_pairs: List[str] = list(text_pair.values())
    mask_idx = rng.integers(0, len(transl_pairs), dtype=int)
    transl_pairs[mask_idx], tgt_txt = noise_fn.compute(transl_pairs[mask_idx], seed)
    src_txt = "</s> ".join(transl_pairs)
    att_mask, labels, input_ids = tokenize(src_txt, tgt_txt, noise_fn, input_max_length, tokenizer)
    return att_mask, labels, input_ids


# def noise_and_tokenize(input_max_length, noise_fn, seed, text, tokenizer):
#     text = text.strip()
#     txt_lst = text.split()
#     ref_len = round(input_max_length - 0.4 * input_max_length)
#     ref_len = len(txt_lst) if len(txt_lst) < ref_len else ref_len
#     text = " ".join(filter(None, txt_lst[0:ref_len]))
#     label_ids, targets = noise_fn.compute(text, seed)
#     input_tok = tokenizer(text, return_special_tokens_mask=False,
#                           add_special_tokens=True, truncation=True,
#                           max_length=input_max_length, padding='longest')
#     out_tok = tokenizer(targets, return_special_tokens_mask=False,
#                         add_special_tokens=True, truncation=True,
#                         max_length=input_max_length, padding='longest',
#                         return_attention_mask=False, return_token_type_ids=False)
#
#     input_ids: List[int] = input_tok['input_ids']
#     att_mask: List[int] = input_tok['attention_mask']
#     labels: List[int] = out_tok['input_ids']
#     labels = [-100 if e == tokenizer.pad_token_id else e for e in labels]
#     return att_mask, labels, input_ids

def noise_and_tokenize(input_max_length, noise_fn, seed, text, tokenizer, ref_len: int = None,
                       tensor_version: bool = False):
    text = text.strip()
    txt_lst = text.split()
    ref_len = round(input_max_length - 0.35 * input_max_length) if ref_len is None else ref_len
    ref_len = len(txt_lst) if len(txt_lst) < ref_len else ref_len
    text = " ".join(filter(None, txt_lst[0:ref_len]))
    inp, targets = noise_fn.compute(text, seed)
    if tensor_version:
        att_mask, labels, input_ids = tokenize_torch(inp, targets, noise_fn, input_max_length, tokenizer)
    else:
        att_mask, labels, input_ids = tokenize(inp, targets, noise_fn, input_max_length, tokenizer)
        att_mask, labels, input_ids = torch.tensor(att_mask), torch.tensor(labels), torch.tensor(input_ids)
    return att_mask, labels, input_ids


def tokenize(inp_text, targets, noise_fn, input_max_length, tokenizer):
    input_tok = tokenizer(inp_text, return_special_tokens_mask=False,
                          add_special_tokens=True, truncation=True,
                          max_length=input_max_length, padding='longest')
    tgt_len = input_max_length
    if noise_fn.n_groups > 1:
        tgt_len = input_max_length // noise_fn.n_groups
    out_tok = tokenizer(targets, return_special_tokens_mask=False,
                        add_special_tokens=True, truncation=True,
                        max_length=tgt_len, padding='longest',
                        return_attention_mask=False, return_token_type_ids=False)
    input_ids: List[int] = input_tok['input_ids']
    att_mask: List[int] = input_tok['attention_mask']
    labels: Union[List[List[int]], List[int]] = out_tok['input_ids']
    if noise_fn.n_groups > 1:
        labels: List[List[int]]
        for g_idx in range(len(labels)):
            group_i = labels[g_idx]
            for i in range(len(group_i)):
                if group_i[i] == tokenizer.pad_token_id or (
                        group_i[i] == tokenizer.eos_token_id and g_idx != (len(labels) - 1)):
                    group_i[i] = -100
        while len(labels) > noise_fn.n_groups:
            del labels[-1]
        while len(labels) < noise_fn.n_groups:
            labels.append([-100 for _ in range(len(labels[0]))])
        assert len(labels) == noise_fn.n_groups
    else:
        labels = [-100 if e == tokenizer.pad_token_id else e for e in labels]
    return att_mask, labels, input_ids


def tokenize_torch(inp_text: str, targets: Union[str, List[str]], noise_fn: Union[MT5NoiseFunction, MT6NoiseFunction],
                   input_max_length: int, tokenizer: MT5TokenizerFast):
    input_tok = tokenizer(inp_text, return_special_tokens_mask=False,
                          add_special_tokens=True, truncation=True,
                          max_length=input_max_length, padding='longest',
                          return_tensors="pt")
    tgt_len = input_max_length
    if noise_fn.n_groups > 1:
        tgt_len = input_max_length // noise_fn.n_groups
    out_tok = tokenizer(targets, return_special_tokens_mask=False,
                        add_special_tokens=True, truncation=True,
                        max_length=tgt_len, padding='longest', return_tensors="pt",
                        return_attention_mask=False, return_token_type_ids=False)
    input_ids: Tensor = input_tok['input_ids']
    att_mask: Tensor = input_tok['attention_mask']
    labels: Tensor = out_tok['input_ids']
    if noise_fn.n_groups > 1:
        if labels.shape[0] > noise_fn.n_groups:
            labels = labels[0:noise_fn.n_groups, :]
        labels_tmp = labels[:-1, :]
        labels_tmp[labels_tmp == tokenizer.eos_token_id] = -100
        # labels[:-1, :] = torch.where(labels[:-1, :] == tokenizer.eos_token_id, -100)
        labels[labels == tokenizer.pad_token_id] = -100

        if labels.shape[0] < noise_fn.n_groups:
            dummy_tensor = torch.empty(noise_fn.n_groups - labels.shape[0], labels.shape[1], dtype=torch.int).fill_(
                -100)
            labels = torch.cat([labels, dummy_tensor], dim=0)

    return att_mask.view(-1), labels, input_ids.view(-1)
