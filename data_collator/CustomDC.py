from typing import Dict, List

import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import MBartTokenizer
from torch.nn.utils.rnn import pad_sequence
from utilities.utility import collate_pad
from custom_datasets.MBartTranslationDataset import MBartTranslationDataset



if __name__ == '__main__':
    translation_ds = load_dataset("wmt14", "fr-en",
                                  cache_dir="D:\\datasets\\wmt14",
                                  split=f"train[0:512]")

    tok_en = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="fr_XX")

    translation_ds = MBartTranslationDataset(translation_ds, tok_en, "fr")

    dl = DataLoader(translation_ds, batch_size=5, drop_last=True, collate_fn=collate_pad)

    for e in dl:
        pass
