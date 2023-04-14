import random
from functools import partial
from typing import List, Set, Tuple

import numpy as np
import torch
from datasets import load_dataset
from numpy.random import Generator
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch import Tensor
from transformers import PreTrainedTokenizer, MT5TokenizerFast
import datasets

from custom_datasets.MT6PreTrainingDataset import MT6PreTrainingDataset
from noise_functions.MT6NoiseFunction import MT6NoiseFunction


class MT6TranslationDataset(Dataset):

    def __init__(self, hugg_dataset: datasets.Dataset, tokenizer: MT5TokenizerFast, trg_lang: str,
                 src_lang: str = 'en', ds_field="translation", input_max_length: int = 128, min_words: int = 4,
                 skip_rows: Set[int] = None, noise_fn: MT6NoiseFunction = MT6NoiseFunction(return_list=True)):
        super(MT6TranslationDataset, self).__init__()

        self.hugg_dataset: datasets.Dataset = hugg_dataset
        self.tokenizer: MT5TokenizerFast = tokenizer
        self.trg_lang: str = trg_lang
        self.src_lang: str = src_lang
        self.ds_field: str = ds_field
        self.input_max_length: int = input_max_length
        self.min_words: int = min_words
        self.skip_rows: Set = set() if skip_rows is None else skip_rows
        self.noise_fn: MT6NoiseFunction = noise_fn

    def __len__(self):
        return len(self.hugg_dataset)

    def __getitem__(self, index):
        sent = self.hugg_dataset[index][self.ds_field]
        src, tgt = sent[self.src_lang], sent[self.trg_lang]

        rng = np.random.default_rng(index)
        if index in self.skip_rows:
            src, tgt = self.retrieve_src_tgt(rng)

        if self.noise_fn is not None:
            tsp_src, targets, tsp_tgt = self.translation_span_corruption(index, rng, src, tgt)
            if self.noise_fn.return_list:
                while len(targets) < self.noise_fn.n_groups:
                    new_src, new_tgt = self.retrieve_src_tgt(rng)
                    src = src + " " + new_src
                    tgt = tgt + " " + new_tgt
                    tsp_src, targets, tsp_tgt = self.translation_span_corruption(index, rng, src, tgt)
            max_len = self.input_max_length // len(targets)
            inputs = tsp_src + " " + tsp_tgt
        else:
            inputs, targets = src, tgt
            max_len = self.input_max_length

        inp_tok = self.tokenizer(inputs, return_special_tokens_mask=False,
                                 add_special_tokens=True, truncation=True,
                                 max_length=self.input_max_length,
                                 return_tensors='pt')

        out_tok = self.tokenizer(targets, return_special_tokens_mask=False,
                                 add_special_tokens=True, truncation=True,
                                 max_length=max_len, padding='longest',
                                 return_tensors='pt', return_attention_mask=False, return_token_type_ids=False)

        labels: Tensor = out_tok['input_ids']
        labels: Tensor = torch.where(labels == 0, -100, labels)
        return {'input_ids': inp_tok['input_ids'].view(-1), 'labels': labels,
                'attention_mask': inp_tok['attention_mask'].view(-1)}

    def translation_span_corruption(self, index: int, rng: Generator, src: str, trg: str) -> Tuple[str, List[str], str]:
        if rng.random() < 0.5:
            src, targets = self.noise_fn.compute(src, index)
        else:
            trg, targets = self.noise_fn.compute(trg, index)
        return src, targets, trg

    def retrieve_src_tgt(self, rng) -> Tuple[str, str]:
        src: str = ""
        tgt: str = ""
        while len(src.split(" ")) < self.min_words:
            index = rng.integers(0, len(self.hugg_dataset) - 1, dtype=int)
            if index in self.skip_rows:
                continue
            sent = self.hugg_dataset[index][self.ds_field]
            src, tgt = sent[self.src_lang], sent[self.trg_lang]
        return src, tgt


if __name__ == '__main__':
    from tqdm import tqdm
    from utilities.utility import collate_pad

    translation_ds = load_dataset("wmt14", "fr-en",
                                  cache_dir="D:\\datasets\\wmt14",
                                  split=f"train",
                                  verification_mode='no_checks')
    pre_train_ds = load_dataset("text", data_files={"train": ["D:\\datasets\\test_hugg_en\\test_data_hugg.txt"]},
                                cache_dir="D:\\datasets\\test_hugg_en", split=f'train',
                                verification_mode='no_checks')

    #translation_ds = translation_ds.with_format("pt")

    tok_en = MT5TokenizerFast.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5")
    # tok_fr = MT5TokenizerFast.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5")

    en_fr_ds = MT6TranslationDataset(translation_ds, tok_en, src_lang="en", trg_lang="fr")
    pre_train_ds = MT6PreTrainingDataset(pre_train_ds, tok_en)
    # fr_en_ds = MT6TranslationDataset(translation_ds, tok_fr, src_lang="fr", trg_lang="en")

    for e in tqdm(DataLoader(ConcatDataset([en_fr_ds, pre_train_ds]), shuffle=True, batch_size=1024,
                             collate_fn=partial(collate_pad, pad_token_id=tok_en.pad_token_id))):
        pass
