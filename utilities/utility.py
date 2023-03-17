from typing import List, Dict, Tuple

import torch
from datasets import Dataset
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad


def collate_pad(batch: List[Dict[str, Tensor]], pad_token_id: int) -> Dict[str, Tensor]:
    inp_ids_list: List[Tensor] = []
    labels_list: List[Tensor] = []
    att_mask_list: List[Tensor] = []
    elem: Dict[str, Tensor]
    for elem in batch:
        inp_ids_list.append(elem['input_ids'])
        labels_list.append(elem['labels'])
        att_mask_list.append(elem['attention_mask'])

    max_len_labels = max(labels_list, key=lambda e: e.shape[-1]).shape[-1]
    padded_inp: Tensor = pad_sequence(inp_ids_list, True, padding_value=pad_token_id)
    for i in range(len(labels_list)):
        t: Tensor = labels_list[i]
        labels_list[i] = pad(t, pad=(0, max_len_labels - t.shape[-1]), mode="constant", value=-100)
    padded_lab = torch.stack(labels_list)
    padded_att: Tensor = pad_sequence(att_mask_list, True, padding_value=0)
    # tgt_len = max([padded_inp.shape[-1], padded_lab.shape[-1], padded_att.shape[-1]])
    # padded_inp: Tensor = pad(padded_inp, pad=(0, tgt_len - padded_inp.shape[-1], 0, 0), mode='constant',
    #                          value=pad_token_id)
    # padded_att: Tensor = pad(padded_att, pad=(0, tgt_len - padded_att.shape[-1], 0, 0), mode='constant', value=0)
    # padded_lab: Tensor = pad(padded_lab, pad=(0, tgt_len - padded_lab.shape[-1], 0, 0), mode='constant', value=-100)

    # padded_lab = pad(padded_lab, pad=(0, 10, 0, 0), mode='constant', value=1)
    return {"input_ids": padded_inp, "labels": padded_lab, "attention_mask": padded_att}


def retrieve_lang(ds_indexes: Dict[str, Tuple[int, int]], index: int) -> Tuple[str, int, int]:
    lang: str
    for lang in ds_indexes:
        i_s, i_e = ds_indexes[lang]
        if i_s <= index < i_e:
            return lang, i_s, i_e
    raise RuntimeError(f"Language not found for index {index}")


def prepare_ds_indexes(hugg_datasets: Dict[str, Dataset]) -> Dict[str, Tuple[int, int]]:
    ds_indexes: Dict[str, Tuple[int, int]] = dict()
    last_idx = 0
    for elem in hugg_datasets:
        ds_len = len(hugg_datasets[elem])
        ds_indexes[elem] = (last_idx, last_idx + ds_len)
        last_idx += ds_len
    return ds_indexes
