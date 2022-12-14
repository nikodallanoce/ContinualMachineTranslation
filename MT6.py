import torch
from torch import nn, Tensor
from torch.cuda.amp import GradScaler
from transformers import MT5ForConditionalGeneration, MT5Config
from torch.optim import Optimizer
from tqdm import tqdm


class MT6(nn.Module):

    def __init__(self, mT5_config: MT5Config):
        super(MT6, self).__init__()
        self.dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model: MT5ForConditionalGeneration = MT5ForConditionalGeneration(mT5_config).to(self.dev)
