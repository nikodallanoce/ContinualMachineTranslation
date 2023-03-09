from typing import Dict, List

import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import MBartTokenizer
from torch.nn.utils.rnn import pad_sequence

from custom_datasets.MBartTranslationDataset import MBartTranslationDataset


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


if __name__ == '__main__':
    translation_ds = load_dataset("wmt14", "fr-en",
                                  cache_dir="D:\\datasets\\wmt14",
                                  split=f"train[0:512]")

    tok_en = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="fr_XX")

    translation_ds = MBartTranslationDataset(translation_ds, tok_en, "fr")

    dl = DataLoader(translation_ds, batch_size=5, drop_last=True, collate_fn=collate_pad)

    for e in dl:
        pass
