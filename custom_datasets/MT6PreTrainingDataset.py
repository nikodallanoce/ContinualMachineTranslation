import random
from typing import Dict, Tuple

import datasets
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, MT5Tokenizer

from noise_functions.MT6NoiseFunction import MT6NoiseFunction
from utilities.utility import prepare_ds_indexes, retrieve_lang


class MT6PreTrainingDataset(Dataset):

    def __init__(self, hugg_dataset: datasets.Dataset, tokenizer: MT5Tokenizer,
                 input_max_length=128, ds_field: str = "text",
                 noise_fn: MT6NoiseFunction = MT6NoiseFunction(return_list=True)):
        super().__init__()
        # self.ds_indexes: Dict[str, Tuple[int, int]] = prepare_ds_indexes(hugg_datasets)
        self.dataset: datasets.Dataset = hugg_dataset  # datasets.concatenate_datasets(list(hugg_datasets.values()))
        self.tokenizer: MT5Tokenizer = tokenizer
        self.input_max_length: int = input_max_length
        self.ds_field = ds_field
        self.noise_fn = noise_fn

    def __len__(self):
        return len(self.dataset)

    # def __getitem__(self, index):
    #     text = self.hugg_dataset[index]['text']
    #     new_index = index
    #
    #     while len(text) < 5:
    #         new_index = random.randint(0, self.hugg_dataset.num_rows - 1)
    #         text = self.hugg_dataset[new_index]['text']
    #
    #     source, target = self.noise_fn.compute_for_mt5(text, new_index, noise_density=0.5)
    #     tokenized = self.tokenizer(source, text_target=target, return_special_tokens_mask=False,
    #                                add_special_tokens=True, truncation=True,
    #                                max_length=self.input_max_length, #padding='max_length',
    #                                return_tensors='pt')
    #
    #     input_ids: Tensor = tokenized['input_ids'].view(-1)
    #     att_mask: Tensor = tokenized['attention_mask'].view(-1)
    #     labels: Tensor = tokenized['labels'].view(-1)
    #     labels: Tensor = torch.where(labels == 0, -100, labels)
    #
    #     outputs = {"input_ids": input_ids, "labels": labels, "attention_mask": att_mask}
    #
    #     return outputs

    def __getitem__(self, index):
        field = self.ds_field
        sampled_sent_min_len = 4
        text = self.dataset[index][field]

        rng = np.random.default_rng(index)
        while len(text.split(" ")) < sampled_sent_min_len:
            new_index = rng.integers(0, len(self.dataset) - 1, dtype=int)
            text = self.dataset[new_index][field]

        n_groups = 3
        text, targets = self.noise_fn.compute(text, index)
        while len(targets) < n_groups:  # only if return_list = True
            new_index =  rng.integers(0, len(self.dataset) - 1, dtype=int)
            text = self.dataset[new_index][field]
            text, targets = self.noise_fn.compute(text, new_index)

        input_tok = self.tokenizer(text, return_special_tokens_mask=False,
                              add_special_tokens=True, truncation=True,
                              max_length=self.input_max_length, padding='longest',
                              return_tensors='pt')
        # while len(targets) < 3:
        #     targets.append("")

        out_tok = self.tokenizer(targets, return_special_tokens_mask=False,
                            add_special_tokens=True, truncation=True,
                            max_length=self.input_max_length // len(targets), padding='longest',
                            return_tensors='pt', return_attention_mask=False, return_token_type_ids=False)

        input_ids: Tensor = input_tok['input_ids'].view(-1)
        att_mask: Tensor = input_tok['attention_mask'].view(-1)
        labels: Tensor = out_tok['input_ids']
        labels: Tensor = torch.where(labels == 0, -100, labels)

        outputs = {"input_ids": input_ids, "labels": labels, "attention_mask": att_mask}

        return outputs
