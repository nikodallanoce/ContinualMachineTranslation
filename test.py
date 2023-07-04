import torch
from datasets import load_dataset, concatenate_datasets, interleave_datasets
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler, RandomSampler

if __name__ == '__main__':
    ds1 = load_dataset("yhavinga/ccmatrix", "en-fr",
                       cache_dir=f"/data/n.dallanoce/cc_en_fr",
                       split=f"train[0:10]",
                       verification_mode='no_checks')
    ds2 = load_dataset("yhavinga/ccmatrix", "en-de",
                       cache_dir=f"/data/n.dallanoce/cc_en_de",
                       split=f"train[0:10]",
                       verification_mode='no_checks')

    concat_ds = ConcatDataset([ds1, ds2])
    for ds in concat_ds.datasets:
        pass

    dl = DataLoader(concat_ds, shuffle=True)
    for e in dl:
        print(e['translation'].keys())
        pass
