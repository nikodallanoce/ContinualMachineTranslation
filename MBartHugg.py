from typing import List
from accelerate import Accelerator, find_executable_batch_size
import torch
from torch import nn, Tensor
from torch.cuda.amp import GradScaler
from transformers import MBartModel, MBartTokenizer, MBartConfig, MBartForConditionalGeneration
from torch.optim import Optimizer
from tqdm import tqdm


class MBart(nn.Module):

    def __init__(self, mBart_config: MBartConfig):
        super(MBart, self).__init__()
        self.model: MBartForConditionalGeneration = MBartForConditionalGeneration(mBart_config)
        # model.to(device)

    def __call__(self, arg):
        return self.model(arg)
