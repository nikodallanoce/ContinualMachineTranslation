import random
from typing import List
import torch

from torch.utils.data.dataset import Dataset, T_co, ConcatDataset, Subset
import numpy as np


def get_buffer(prev_exp_ds: List[Dataset], total_cumulative_size: int, buff_size_frac: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    buffer_rows = round(buff_size_frac * total_cumulative_size)
    subset_rows = [rng.integers(0, len(ds), buffer_rows) for ds in prev_exp_ds]
    rows_per_ds = round(buffer_rows / len(prev_exp_ds))
    buffer = []
    for i in range(len(prev_exp_ds)):
        ds = prev_exp_ds[i]
        ds_idx = subset_rows[i]
        buffer.append(Subset(ds, ds_idx[:rows_per_ds]))
    buffer = ConcatDataset(buffer)
    return buffer


class RandomReplayDataset(Dataset):

    def __init__(self, curr_exp_ds_lst: List[Dataset], prev_exp_ds_lst: List[Dataset], prev_exp_frac: float = 0.05,
                 seed: int = 0):
        curr_exp_ds = ConcatDataset(curr_exp_ds_lst)
        prev_exp_ds = ConcatDataset(prev_exp_ds_lst)
        random.seed(seed)
        prev_exp_idx = [random.randint(0, len(prev_exp_ds)) for _ in
                        range(round(len(curr_exp_ds) * prev_exp_frac))]

        buffer = torch.utils.data.Subset(prev_exp_ds, prev_exp_idx)
        # ds = [reply_memory]
        # ds = ds + curr_exp_ds
        self.dataset = ConcatDataset(curr_exp_ds_lst + [buffer])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> T_co:
        return self.dataset[index]
