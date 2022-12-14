import torch
from torch import nn, Tensor
from torch.cuda.amp import GradScaler
from transformers import MBartModel, MBartTokenizer, MBartConfig, MBartForConditionalGeneration
from torch.optim import Optimizer
from tqdm import tqdm


class MBart(nn.Module):

    def __init__(self, mBart_config: MBartConfig):
        super(MBart, self).__init__()
        self.dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.model: MBartForConditionalGeneration = MBartForConditionalGeneration(mBart_config).to(self.dev)

    def fit(self, train_loader, optimizer, epochs, val_loader=None):
        scaler = GradScaler()
        self.model.train(True)
        for i in range(epochs):
            print(f"Epoch {i + 1} of {epochs}")
            epoch_loss = 0
            eta = tqdm(train_loader)
            for n_batch, batch in enumerate(eta):
                loss = self.training_step(batch, optimizer, scaler, n_batch, 32).item()
                #self.evaluate(batch)
                epoch_loss += loss
                eta.set_postfix(loss=str(epoch_loss / (n_batch + 1))[0:6])

    def training_step(self, batch: list[torch.Tensor], optimizer: Optimizer, scaler: GradScaler, batch_id: int,
                      acc_step: int) -> Tensor:
        # input_ids, att_mask = batch[0]

        input_ids, att_mask, masked_ids = batch
        input_ids: Tensor = input_ids.to(self.dev)
        att_mask: Tensor = att_mask.to(self.dev)
        masked_ids: Tensor = masked_ids.to(self.dev)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            loss = self.model(masked_ids, att_mask, labels=input_ids).loss
            # loss = loss / acc_step

        scaler.scale(loss).backward()  # loss.backward()
        # if (batch_id + 1) % acc_step == 0:
        scaler.step(optimizer)  # optimizer.step()
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return loss

    def evaluate(self, batch: list[torch.Tensor], mask_id: int = 250026):
        input_ids, att_mask, masked_ids = batch
        input_ids: Tensor = input_ids.to(self.dev)
        att_mask: Tensor = att_mask.to(self.dev)
        masked_ids: Tensor = masked_ids.to(self.dev)
        with torch.no_grad():
            output = self.model(masked_ids, att_mask, labels=input_ids)
            loss = output.loss
            # logits = output.logits.softmax(dim=0).topk(1).indices
            # logits = logits.view(logits.shape[0], -1)
            # index_mask = (masked_ids == mask_id).nonzero(as_tuple=False)
            # for mask_index in index_mask:
            #    a = logits[mask_index[0], mask_index[1]]
            #    b = index_mask[mask_index[0], mask_index[1]]
            #    if a==b:
            #        acc = acc + 1
            # acc = acc/index_mask.shape[0]
        return loss
