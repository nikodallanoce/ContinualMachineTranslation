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

    def __init__(self, hugg_dataset: datasets.Dataset, tokenizer: MBartTokenizer, tgt_lang: str,
                 src_lang: str = 'en', ds_field="translation", input_max_length: int = 128, min_words: int = 4,
                 skip_rows: Set[int] = None):
        super(MBartTranslationDataset, self).__init__()

        self.hugg_dataset: datasets.Dataset = hugg_dataset
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.tgt_lang: str = tgt_lang
        self.src_lang: str = src_lang
        self.ds_field: str = ds_field
        self.input_max_length: int = input_max_length
        self.min_words: int = min_words
        self.skip_rows: Set = set() if skip_rows is None else skip_rows

    def __len__(self):
        return len(self.hugg_dataset)

    def __getitem__(self, index):
        sent = self.hugg_dataset[index][self.ds_field]
        src, trg = sent[self.src_lang], sent[self.tgt_lang]

        rng = np.random.default_rng(index)
        while len(src.split(" ")) < self.min_words or index in self.skip_rows:
            index = rng.integers(0, len(self.hugg_dataset) - 1, dtype=int)
            sent = self.hugg_dataset[index][self.ds_field]
            src, trg = sent[self.src_lang], sent[self.tgt_lang]

        outputs = self.tokenizer(src, text_target=trg, return_special_tokens_mask=False,
                                 add_special_tokens=True, truncation=True,
                                 max_length=self.input_max_length,
                                 return_tensors='pt')
        labels = outputs['labels'].view(-1)
        labels = torch.where(labels == 1, -100, labels)  # ancora necessario?
        return {'input_ids': outputs['input_ids'].view(-1), 'labels': labels,
                'attention_mask': outputs['attention_mask'].view(-1)}

# if __name__ == '__main__':
#     from tqdm import tqdm
#     from torch.utils.data import ConcatDataset
#
#     translation_ds = load_dataset("wmt14", "fr-en",
#                                   cache_dir="D:\\datasets\\wmt14",
#                                   split=f"train[0:4096]",
#                                   verification_mode='no_checks')
#
#     translation_ds = translation_ds.with_format("pt")
#
#     tok_en = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", lang1="en_XX", lang2="fr_XX")
#     tok_fr = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", lang1="fr_XX", lang2="en_XX")
#
#     en_fr_ds = MT6TranslationDataset(translation_ds, tok_en, lang1="en", trg_lang="fr")
#     fr_en_ds = MT6TranslationDataset(translation_ds, tok_fr, lang1="fr", trg_lang="en")
#
#     for e in tqdm(DataLoader(ConcatDataset([en_fr_ds, fr_en_ds]), shuffle=True)):
#         pass
