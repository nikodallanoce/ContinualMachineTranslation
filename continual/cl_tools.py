from typing import Tuple, List

import math
from clearml import Dataset
from torch.utils.data import Sampler, BatchSampler, Subset, ConcatDataset
import numpy as np


def get_buffer(prev_exp_ds: List[Dataset], total_cumulative_size: int, buff_size_frac: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    buffer_rows = round(buff_size_frac * total_cumulative_size)
    subset_rows = [rng.integers(0, len(ds), buffer_rows) for ds in prev_exp_ds]
    subset_rows = [r.tolist() for r in subset_rows]
    rows_per_ds = round(buffer_rows / len(prev_exp_ds))
    buffer = []
    for i in range(len(prev_exp_ds)):
        ds = prev_exp_ds[i]
        ds_idx = subset_rows[i]
        buffer.append(Subset(ds, ds_idx[:rows_per_ds]))
    buffer = ConcatDataset(buffer)
    return buffer


class CLSampler(BatchSampler):

    def __init__(self, samplers: Tuple[Sampler[int], Sampler[int]], curr_exp_frac: float, batch_size: int = 1,
                 drop_last: bool = True) -> None:
        super().__init__(None, batch_size, drop_last)
        self.curr_exp_frac = curr_exp_frac
        self.dataset_samp, self.buff_samp = samplers
        if len(self.dataset_samp) <= len(self.buff_samp):
            raise ValueError("len(samplers[0]) <= len(samplers[1])")
        self.curr_exp_ds_samples = round(curr_exp_frac * batch_size)
        self.buff_ds_samples = batch_size - self.curr_exp_ds_samples
        self.curr_exp_len = len(self.dataset_samp)

    def clone(self, batch_size: int, drop_last: bool):
        return CLSampler((self.dataset_samp, self.buff_samp), self.curr_exp_frac, batch_size, drop_last)

    def __iter__(self):
        iter_sampler = [iter(self.dataset_samp), iter(self.buff_samp)]
        ds_samples = [self.curr_exp_ds_samples, self.buff_ds_samples]
        largest_sampler_exhausted = False
        while not largest_sampler_exhausted:
            batch = []
            for i in range(len(iter_sampler)):
                elem = ds_samples[i]
                remaining_elem = elem
                while remaining_elem > 0:
                    try:
                        idx = next(iter_sampler[i]) if i == 0 else self.curr_exp_len + next(
                            iter_sampler[i])
                        batch.append(idx)
                    except StopIteration:
                        if i == 0:
                            largest_sampler_exhausted = True
                            if len(batch) == 0:
                                return
                            break
                        else:
                            iter_sampler[i] = iter(self.buff_samp)
                            idx = self.curr_exp_len + next(iter_sampler[i])
                            batch.append(idx)
                    remaining_elem = remaining_elem - 1
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return self.curr_exp_len // self.curr_exp_ds_samples  # type: ignore[arg-type]
        else:
            return math.ceil(self.curr_exp_len / self.curr_exp_ds_samples)
