from typing import List, Set, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch import Tensor
from transformers import MBartTokenizer
import datasets
from noise_functions.MBartNoiseFunction import MBartNoiseFunction
from utilities.utility import prepare_ds_indexes, retrieve_lang


class MBartPreTrainingDataset(Dataset):

    def __init__(self, hugg_dataset: datasets.Dataset, tokenizer: MBartTokenizer,
                 input_max_length=128, ds_field: str = "text"):
        super(MBartPreTrainingDataset, self).__init__()
        # self.ds_indexes: Dict[str, Tuple[int, int]] = prepare_ds_indexes(hugg_datasets)
        self.dataset: datasets.Dataset = hugg_dataset  # datasets.concatenate_datasets(list(hugg_datasets.values()))
        self.tokenizer: MBartTokenizer = tokenizer
        self.input_max_length: int = input_max_length
        self.noise_fn = MBartNoiseFunction()
        self.ds_field: str = ds_field
        self.ref_len: float = input_max_length - input_max_length * 0.4

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        field = self.ds_field
        sampled_sent_min_len = 4
        label_ids = self.dataset[index][field]

        rng = np.random.default_rng(index)
        while len(label_ids.split(" ")) < sampled_sent_min_len:
            new_index = rng.integers(0, len(self.dataset) - 1, dtype=int)
            label_ids = self.dataset[new_index][field]

        label_ids = label_ids.strip()
        while len(label_ids.split(" ")) < self.ref_len:
            new_index = rng.integers(0, len(self.dataset) - 1, dtype=int)
            new_sent = self.dataset[new_index][field]
            if len(new_sent) < sampled_sent_min_len:
                continue
            label_ids += self.tokenizer.eos_token + " " + new_sent.strip()

        label_ids, masked_ids = self.noise_fn.compute(label_ids, index)
        tokenized = self.tokenizer([label_ids, masked_ids], return_special_tokens_mask=False,
                                   add_special_tokens=True, truncation=True, return_attention_mask=False,
                                   max_length=self.input_max_length, padding='longest',
                                   return_tensors='pt')['input_ids']

        label_ids = tokenized[0].view(-1)
        masked_ids = tokenized[1].view(-1)

        att_mask = torch.where(masked_ids != self.tokenizer.pad_token_id, 1, 0)
        label_ids = torch.where(label_ids == self.tokenizer.pad_token_id, -100, label_ids)
        return {'input_ids': masked_ids, 'labels': label_ids, 'attention_mask': att_mask}
