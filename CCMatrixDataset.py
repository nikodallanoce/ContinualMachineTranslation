from typing import Iterator

import torch
from torch.utils import data
from torch.utils.data.dataset import T_co


class CCMatrixDataset(data.IterableDataset):
    def __iter__(self) -> Iterator[T_co]:
        pass

    def __getitem__(self, index) -> T_co:
        pass