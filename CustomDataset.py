from torch.utils.data import Dataset
from torch import Tensor


class CustomDataset(Dataset):

    def __init__(self, source, target, att_mask):
        super(CustomDataset, self).__init__()
        self.source: Tensor = source
        self.target: Tensor = target
        self.att_mask: Tensor = att_mask

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return (self.source[index], self.att_mask[index]), self.target[index]
