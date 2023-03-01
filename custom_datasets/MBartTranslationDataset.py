from typing import List, Set, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torch.utils.data.dataset import T_co
from transformers import PreTrainedTokenizer, MBartTokenizer
import datasets


class MBartTranslationDataset(Dataset):

    def __init__(self, hugg_dataset: datasets.Dataset, tokenizer: MBartTokenizer, trg_lang: str,
                 src_lang: str = 'en', ds_field="translation", input_max_length: int = 128):
        super(MBartTranslationDataset, self).__init__()

        self.hugg_dataset: datasets.Dataset = hugg_dataset
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.trg_lang: str = trg_lang
        self.src_lang: str = src_lang
        self.ds_field: str = ds_field
        self.input_max_length: int = input_max_length

    def __len__(self):
        return len(self.hugg_dataset)

    def __getitem__(self, index):
        sent = self.hugg_dataset[index][self.ds_field]
        src, trg = sent[self.src_lang], sent[self.trg_lang]
        outputs = self.tokenizer(src, text_target=trg, add_special_tokens=True, padding="max_length", truncation=True,
                                 max_length=self.input_max_length, return_tensors='pt')
        return outputs


if __name__ == '__main__':
    translation_ds = load_dataset("wmt14", "fr-en",
                                  cache_dir="C:\\Users\\dllni\\PycharmProjects\\pretraining\\wmt14",
                                  split=f"train[0:1000]")

    tok_en = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="fr_XX")

    translation_ds = MBartTranslationDataset(translation_ds, tok_en, "fr")

    for e in DataLoader(translation_ds):
        pass
