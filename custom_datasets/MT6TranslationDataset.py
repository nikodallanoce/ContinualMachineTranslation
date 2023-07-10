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
from eval.bleu_utility import get_langs_token
from MT6TokenizerFast import MT6TokenizerFast
from custom_datasets.MT6PreTrainingDataset import MT6PreTrainingDataset, tokenize_torch
from noise_functions.MT5NoiseFunction import MT5NoiseFunction, MT6NoiseFunction

PREFIX_TASK = {'en': "English", 'fr': "French", 'de': "German", 'es': "Spanish"}


class MT6TranslationDataset(Dataset):

    def __init__(self, hugg_dataset: datasets.Dataset, tokenizer: MT5TokenizerFast, tgt_lang: str,
                 src_lang: str = "en", ds_field="translation", input_max_length: int = 128, min_words: int = 4,
                 skip_rows: Set[int] = None, noise_fn=MT6NoiseFunction(pnat=True)):
        super(MT6TranslationDataset, self).__init__()

        self.hugg_dataset: datasets.Dataset = hugg_dataset

        if isinstance(tokenizer, MT6TokenizerFast):
            tok_src_lang, tok_tgt_lang = get_langs_token(src_lang, tgt_lang)
            tokenizer = MT6TokenizerFast.from_pretrained(tokenizer.name_or_path, src_lang=tok_src_lang,
                                                         tgt_lang=tok_tgt_lang)
        self.tokenizer: MT5TokenizerFast = tokenizer
        self.tgt_lang: str = tgt_lang
        self.src_lang: str = src_lang
        self.ds_field: str = ds_field
        self.input_max_length: int = input_max_length
        self.min_words: int = min_words
        self.skip_rows: Set = set() if skip_rows is None else skip_rows
        self.noise_fn: MT6NoiseFunction = noise_fn
        self.labels_name: str = "labels" if noise_fn is None else "labels_tsc"

    def __len__(self):
        return len(self.hugg_dataset)

    def __getitem__(self, index):
        sent = self.hugg_dataset[index][self.ds_field]
        src, tgt = sent[self.src_lang], sent[self.tgt_lang]

        rng = np.random.default_rng(index)
        if index in self.skip_rows:
            src, tgt = self.retrieve_src_tgt(rng)

        if self.noise_fn is not None:
            # src, tgt = self.append_sentences(src, tgt, " ", rng) #concatenate sentence pairs on pre-training
            tsc_src, tsc_tgt = self.translation_span_corruption(src, tgt, index, rng)
            # if self.noise_fn.return_list:
            #     while len(tsc_src) < self.noise_fn.n_groups:
            #         new_src, new_tgt = self.retrieve_src_tgt(rng)
            #         src = src + " " + new_src
            #         tgt = tgt + " " + new_tgt
            #         tsc_src, tsc_tgt = self.translation_span_corruption(src, tgt, index, rng)
            att_mask, labels, input_ids = tokenize_torch(tsc_src, tsc_tgt, self.noise_fn, self.input_max_length,
                                                         self.tokenizer)
            # att_mask, labels, input_ids = torch.tensor(att_mask), torch.tensor(labels), torch.tensor(input_ids)
        else:
            src, tgt = self.append_sentences(src, tgt, " </s> ", rng)
            if not isinstance(self.tokenizer, MT6TokenizerFast):
                src = f"translate {PREFIX_TASK[self.src_lang]} to {PREFIX_TASK[self.tgt_lang]}: " + src
            inputs, targets = src, tgt

            inp_tok = self.tokenizer(inputs, return_special_tokens_mask=False,
                                     add_special_tokens=True, truncation=True,
                                     max_length=self.input_max_length,
                                     return_tensors='pt')

            out_tok = self.tokenizer(text_target=targets, return_special_tokens_mask=False,
                                     add_special_tokens=True, truncation=True,
                                     max_length=self.input_max_length, padding='longest',
                                     return_tensors='pt', return_attention_mask=False, return_token_type_ids=False)

            input_ids = inp_tok['input_ids'].view(-1)
            att_mask: Tensor = inp_tok['attention_mask'].view(-1)
            labels: Tensor = out_tok['input_ids'].view(-1)
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {'input_ids': input_ids, self.labels_name: labels,
                'attention_mask': att_mask}

    def append_sentences(self, src: str, tgt: str, eos_sep: str, rng):
        while len(src.split(" ")) < (self.input_max_length - self.input_max_length * 0.35):
            app_src, app_tgt = self.retrieve_src_tgt(rng)
            if len(app_src.split()) > 3 and len(app_tgt.split()) > 3:
                src = src + eos_sep + app_src
                tgt = tgt + eos_sep + app_tgt
        return src, tgt

    def translation_span_corruption(self, src: str, trg: str, index: int, rng: Generator) -> Tuple[str, List[str]]:
        transl_pairs: List[str] = [src, trg]
        mask_idx = rng.integers(0, len(transl_pairs), dtype=int)
        transl_pairs[mask_idx], tgt_txt = self.noise_fn.compute(transl_pairs[mask_idx], index)
        src_txt = f"{self.tokenizer.eos_token_id} ".join(transl_pairs)
        return src_txt, tgt_txt

    def retrieve_src_tgt(self, rng) -> Tuple[str, str]:
        src: str = ""
        tgt: str = ""
        while len(src.split(" ")) < self.min_words:
            index = rng.integers(0, len(self.hugg_dataset), dtype=int)
            if index not in self.skip_rows:
                sent = self.hugg_dataset[index][self.ds_field]
                src, tgt = sent[self.src_lang], sent[self.tgt_lang]
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

    # translation_ds = translation_ds.with_format("pt")

    tok_en = MT5TokenizerFast.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5")
    # tok_fr = MT5TokenizerFast.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5")

    en_fr_ds = MT6TranslationDataset(translation_ds, tok_en, src_lang="en", tgt_lang="fr")
    pre_train_ds = MT6PreTrainingDataset(pre_train_ds, tok_en)
    # fr_en_ds = MT6TranslationDataset(translation_ds, tok_fr, lang1="fr", trg_lang="en")

    for e in tqdm(DataLoader(ConcatDataset([en_fr_ds, pre_train_ds]), shuffle=True, batch_size=1024,
                             collate_fn=partial(collate_pad, pad_token_id=tok_en.pad_token_id))):
        pass
