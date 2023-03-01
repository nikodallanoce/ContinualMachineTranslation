import random

import datasets
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
        label_ids = self.hugg_dataset[index]['text']
        new_index = index
        while len(label_ids) < 5:
            new_index = random.randint(0, self.hugg_dataset.num_rows - 1)
            label_ids = self.hugg_dataset[new_index]['text']

        label_ids, targets = self.noise_fn.compute(label_ids, new_index)
        # targets.append(label_ids)
        input_ids = self.tokenizer(label_ids, return_special_tokens_mask=False,
                                   add_special_tokens=True, truncation=True,
                                   max_length=self.input_max_length, padding='max_length',
                                   return_tensors='pt')['input_ids']
        targets = self.tokenizer(targets, return_special_tokens_mask=False,
                                 add_special_tokens=True, truncation=True,
                                 max_length=self.input_max_length, padding='max_length',
                                 return_tensors='pt')['input_ids']

        #zeros = torch.zeros_like(input_ids)
        #targets = targets.view(1, -1)
        #zeros[:, :targets.shape[1]] = targets
        #targets = zeros
        outputs = {"input_ids": input_ids, "labels": targets}

        return outputs
