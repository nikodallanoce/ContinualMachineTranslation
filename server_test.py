import datasets
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from datasets import load_dataset
import pandas as pd

if __name__ == '__main__':
    ds = load_dataset("text", data_files={"train": ["/data/n.dallanoce/cc100/en.txt"]}, cache_dir="/data/n.dallanoce/cc100/hugg_en", split='train')
    ds = ds.with_format("torch")
    ds = DataLoader(ds)
    for e in ds:
        print()
    print()
