from typing import List, Dict

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

    padded_inp: Tensor = pad_sequence(inp_ids_list, True, padding_value=pad_token_id)
    padded_lab: Tensor = pad_sequence(labels_list, True, padding_value=-100)
    padded_att: Tensor = pad_sequence(att_mask_list, True, padding_value=0)
    tgt_len = max([padded_inp.shape[-1], padded_lab.shape[-1], padded_att.shape[-1]])
    # padded_inp: Tensor = pad(padded_inp, pad=(0, tgt_len - padded_inp.shape[-1], 0, 0), mode='constant',
    #                          value=pad_token_id)
    # padded_att: Tensor = pad(padded_att, pad=(0, tgt_len - padded_att.shape[-1], 0, 0), mode='constant', value=0)
    # padded_lab: Tensor = pad(padded_lab, pad=(0, tgt_len - padded_lab.shape[-1], 0, 0), mode='constant', value=-100)

    #padded_lab = pad(padded_lab, pad=(0, 10, 0, 0), mode='constant', value=1)
    return {"input_ids": padded_inp, "labels": padded_lab, "attention_mask": padded_att}
