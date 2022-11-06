import torch
from torch import nn, Tensor
from transformers import MBartModel, MBartTokenizer, MBartConfig, MBartForConditionalGeneration
from torch.optim import Optimizer
from tqdm import tqdm


class MBart(nn.Module):

    def __init__(self, mBart_config: MBartConfig):
        super(MBart, self).__init__()
        self.dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.model: MBartForConditionalGeneration = MBartForConditionalGeneration(mBart_config).to(self.dev)

    def fit(self, dataloader, optimizer, epochs):
        for i in range(epochs):
            print(f"Epoch {i}")
            epoch_loss = 0
            eta = tqdm(dataloader)
            for n_batch, batch in enumerate(eta):
                loss = self.training_step(batch, optimizer).item()
                epoch_loss += loss
                eta.set_postfix(loss=str(epoch_loss / (n_batch + 1))[0:6])

    def training_step(self, batch, optimizer: Optimizer) -> Tensor:
        input_ids, att_mask = batch[0]
        input_ids = input_ids.to(self.dev)
        att_mask = att_mask.to(self.dev)
        masked_ids = batch[1].to(self.dev)

        optimizer.zero_grad(set_to_none=True)
        loss = self.model.forward(input_ids, att_mask, labels=masked_ids).loss
        loss.backward()
        optimizer.step()
        return loss
