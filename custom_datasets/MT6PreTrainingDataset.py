import random
from typing import Dict, Tuple

import datasets
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, MT5Tokenizer

from noise_functions.MT6NoiseFunction import MT6NoiseFunction
from utilities.utility import prepare_ds_indexes, retrieve_lang


class MT6PreTrainingDataset(Dataset):

    def __init__(self, hugg_datasets: Dict[str, datasets.Dataset], tokenizers: Dict[str, MT5Tokenizer],
                 noise_fn: MT6NoiseFunction = MT6NoiseFunction(), input_max_length=128):
        super().__init__()
        self.ds_indexes: Dict[str, Tuple[int, int]] = prepare_ds_indexes(hugg_datasets)
        self.dataset: datasets.Dataset = datasets.concatenate_datasets(list(hugg_datasets.values()))
        self.tokenizers: Dict[str, MT5Tokenizer] = tokenizers
        self.input_max_length: int = input_max_length
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
        field = "text"
        text = self.dataset[index][field]
        text = text.strip()
        lang, i_s, i_e = retrieve_lang(self.ds_indexes, index)
        tokenizer = self.tokenizers[lang]
        ref_len = self.input_max_length - self.input_max_length * 0.4
        new_index = index
        n_groups = 3
        text, targets = self.noise_fn.compute(text, new_index)
        while len(targets) < n_groups:  # only if return_list = True
            new_index = random.randint(0, self.hugg_dataset.num_rows - 1)
            text = self.hugg_dataset[new_index][field]
            ext, targets = self.noise_fn.compute(text, new_index)

        input_tok = tokenizer(text, return_special_tokens_mask=False,
                              add_special_tokens=True, truncation=True,
                              max_length=self.input_max_length, padding='longest',
                              return_tensors='pt')
        # while len(targets) < 3:
        #     targets.append("")

        out_tok = tokenizer(targets, return_special_tokens_mask=False,
                            add_special_tokens=True, truncation=True,
                            max_length=self.input_max_length // len(targets), padding='longest',
                            return_tensors='pt', return_attention_mask=False, return_token_type_ids=False)

        input_ids: Tensor = input_tok['input_ids'].view(-1)
        att_mask: Tensor = input_tok['attention_mask'].view(-1)
        labels: Tensor = out_tok['input_ids']
        labels: Tensor = torch.where(labels == 0, -100, labels)

        outputs = {"input_ids": input_ids, "labels": labels, "attention_mask": att_mask}

        return outputs
