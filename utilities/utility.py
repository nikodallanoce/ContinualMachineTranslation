from enum import Enum
from typing import List, Dict, Tuple

import torch
from datasets import Dataset
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad


def collate_pad_mt6(batch: List[Dict[str, Tensor]], labels: List[str], pad_token_id: int):
    # batch_pnat_lst: List[Dict[str, Tensor]] = []
    # batch_transl_lst: List[Dict[str, Tensor]] = []
    lab_batch: Dict[str, List[Dict[str, Tensor]]] = {}
    for lab_name in labels:
        lab_batch[lab_name] = []

    elem: Dict[str, Tensor]
    for elem in batch:
        if labels[0] in elem:
            lab_batch[labels[0]].append(elem)
        elif labels[1] in elem:
            lab_batch[labels[1]].append(elem)
        else:
            raise KeyError("Provided labels and labels in the batch do not match!")

    return_list = []
    for lab_name, lab_lst in lab_batch.items():
        if len(lab_lst) > 0:
            return_list.append(collate_pad(lab_lst, pad_token_id, labels_name=lab_name))
    return return_list

    # batched_pnat = collate_torch_iterable(batch_pnat_lst, pad_token_id, num_workers=num_workers,
    #                                       labels_name=labels[0])
    # if len(batch_transl_lst) > 0:
    #     batched_transl = collate_torch_iterable(batch_transl_lst, pad_token_id, num_workers=num_workers,
    #                                             labels_name=labels[1])
    #     return [batched_pnat, batched_transl]
    # return [batched_pnat]


def collate_pad(batch: List[Dict[str, Tensor]], pad_token_id: int, labels_name: str = 'labels') -> Dict[str, Tensor]:
    inp_ids_list: List[Tensor] = []
    labels_list: List[Tensor] = []
    att_mask_list: List[Tensor] = []
    elem: Dict[str, Tensor]
    for elem in batch:
        inp_ids_list.append(elem['input_ids'])
        labels_list.append(elem[labels_name])
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
    return {"input_ids": padded_inp, labels_name: padded_lab, "attention_mask": padded_att}


def collate_torch_iterable(batch_list, pad_token_id: int, num_workers: int = 1, labels_name: str = "labels"):
    from concurrent.futures import ThreadPoolExecutor
    batch_size, batch_reminder = divmod(len(batch_list), num_workers)
    results = []
    i_start = 0
    i_end = 0
    with ThreadPoolExecutor(num_workers) as executor:
        for i in range(num_workers):
            i_end = i_end + batch_size + int(batch_reminder > 0)
            batch_reminder = batch_reminder - 1
            results.append(
                executor.submit(parallel_copy, batch_list=batch_list[i_start:i_end], labels_name=labels_name))
            i_start = i_end
    batch_list = []
    for res in results:
        batch_list.extend(res.result())
    return collate_pad(batch_list, pad_token_id, labels_name)


def parallel_copy(batch_list: List, labels_name: str):
    for batch in batch_list:
        batch['input_ids'] = torch.tensor(batch['input_ids'])
        batch['attention_mask'] = torch.tensor(batch['attention_mask'])
        batch[labels_name] = torch.tensor(batch[labels_name])
    return batch_list


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

class TrainingStrategy(Enum):
    PRE_TRAINING = 0,
    FINE_TUNING = 1,
    FINE_TUNING_LANG = 2