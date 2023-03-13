import random

import datasets
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from noise_functions.MT6NoiseFunction import MT6NoiseFunction


class MT6PreTrainingDataset(Dataset):

    def __init__(self, hugg_dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, input_max_length=128):
        super().__init__()
        self.hugg_dataset: datasets.Dataset = hugg_dataset
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.input_max_length: int = input_max_length
        self.noise_fn = MT6NoiseFunction()

    def __len__(self):
        return len(self.hugg_dataset)

    def __getitem__(self, index):
        text = self.hugg_dataset[index]['text']
        new_index = index

        while len(text) < 5:
            new_index = random.randint(0, self.hugg_dataset.num_rows - 1)
            text = self.hugg_dataset[new_index]['text']

        source, target = self.noise_fn.compute_for_mt5(text, new_index, noise_density=0.5)
        tokenized = self.tokenizer(source, text_target=target, return_special_tokens_mask=False,
                                   add_special_tokens=True, truncation=True,
                                   max_length=self.input_max_length, #padding='max_length',
                                   return_tensors='pt')

        input_ids: Tensor = tokenized['input_ids'].view(-1)
        att_mask: Tensor = tokenized['attention_mask'].view(-1)
        labels: Tensor = tokenized['labels'].view(-1)
        labels: Tensor = torch.where(labels == 0, -100, labels)

        outputs = {"input_ids": input_ids, "labels": labels, "attention_mask": att_mask}

        return outputs

    # def __getitem__(self, index):
    #     text = self.hugg_dataset[index]['text']
    #     new_index = index
    #     label_ids, targets = self.noise_fn.compute(text, new_index, return_list=True, noise_density=0.3)
    #     while len(targets) < 3: #only if return_list = True
    #         new_index = random.randint(0, self.hugg_dataset.num_rows - 1)
    #         text = self.hugg_dataset[new_index]['text']
    #         label_ids, targets = self.noise_fn.compute(text, new_index, return_list=True, noise_density=0.3)
    #
    #     input_tok = self.tokenizer(label_ids, return_special_tokens_mask=False,
    #                                add_special_tokens=True, truncation=True,
    #                                max_length=self.input_max_length, padding='max_length',
    #                                return_tensors='pt')
    #     # while len(targets) < 3:
    #     #     targets.append("")
    #
    #     out_tok = self.tokenizer(targets, return_special_tokens_mask=False,
    #                              add_special_tokens=True, truncation=True,
    #                              max_length=self.input_max_length // len(targets), padding='max_length',
    #                              return_tensors='pt', return_attention_mask=False, return_token_type_ids=False)
    #
    #     input_ids: Tensor = input_tok['input_ids'].view(-1)
    #     att_mask: Tensor = input_tok['attention_mask'].view(-1)
    #     labels: Tensor = out_tok['input_ids']
    #     labels: Tensor = torch.where(labels == 0, -100, labels)
    #
    #     outputs = {"input_ids": input_ids, "labels": labels, "attention_mask": att_mask}
    #
    #     return outputs
