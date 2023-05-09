import random
from typing import Iterator, List, Optional
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co
import datasets


class IterMT6PreTrainingDataset(IterableDataset):

    def __init__(self, iterable_datasets: List[datasets.iterable_dataset.IterableDataset],
                 weights: Optional[List[float]] = None) -> None:
        super(IterMT6PreTrainingDataset).__init__()
        iter_datasets: List[Iterator[datasets.iterable_dataset.IterableDataset]] = [iter(ds) for ds in
                                                                                    iterable_datasets]
        self.iter = ChoiceIterator(iter_datasets, weights)

    def __iter__(self) -> Iterator[T_co]:
        return self.iter


class ChoiceIterator(Iterator):

    def __init__(self, iter_datasets: List[Iterator[datasets.iterable_dataset.IterableDataset]],
                 weights: Optional[List[float]] = None) -> None:
        super(ChoiceIterator).__init__()
        self.iter_datasets: List[Iterator[datasets.iterable_dataset.IterableDataset]] = iter_datasets
        self.weights: Optional[List[float]] = weights

    def __next__(self):
        sampled_ds = random.choices(self.iter_datasets, weights=self.weights, k=1)[0]
        return next(sampled_ds)
