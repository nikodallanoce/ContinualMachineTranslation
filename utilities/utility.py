from typing import List, Dict

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def collate_pad(batch: List[Dict[str, Tensor]]):
    inp_ids_list: List[Tensor] = []
    labels_list: List[Tensor] = []
    att_mask_list: List[Tensor] = []
    elem: Dict[str, Tensor]
    for elem in batch:
        inp_ids_list.append(elem['input_ids'])
        labels_list.append(elem['labels'])
        att_mask_list.append(elem['attention_mask'])

    padded_inp = pad_sequence(inp_ids_list, True)
    padded_lab = pad_sequence(labels_list, True, padding_value=-100)
    padded_att = pad_sequence(att_mask_list, True)
    return {"input_ids": padded_inp, "labels": padded_lab, "attention_mask": padded_att}
