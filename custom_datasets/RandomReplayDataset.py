import random
from typing import List
import torch

from torch.utils.data.dataset import Dataset, T_co, ConcatDataset
import numpy as np


class RandomReplayDataset(Dataset):

    def __init__(self, curr_exp_ds_lst: List[Dataset], prev_exp_ds_lst: List[Dataset], prev_exp_percentage: float = 0.05):
        curr_exp_ds = ConcatDataset(curr_exp_ds_lst)
        prev_exp_ds = ConcatDataset(prev_exp_ds_lst)
        prev_exp_idx = [random.randint(0, len(prev_exp_ds)) for _ in
                        range(round(len(curr_exp_ds) * prev_exp_percentage))]

        reply_memory = torch.utils.data.Subset(prev_exp_ds, prev_exp_idx)
        # ds = [reply_memory]
        # ds = ds + curr_exp_ds
        self.dataset = ConcatDataset(curr_exp_ds_lst + [reply_memory])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> T_co:
        return self.dataset[index]
