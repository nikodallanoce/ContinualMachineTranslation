from typing import List
from accelerate import Accelerator, find_executable_batch_size
import torch
from torch import nn, Tensor
from torch.cuda.amp import GradScaler
from transformers import MBartModel, MBartTokenizer, MBartConfig, MBartForConditionalGeneration
from torch.optim import Optimizer
from tqdm import tqdm


class MBart(nn.Module):

    def __init__(self, mBart_config: MBartConfig, device_ids: List[int]):
        super(MBart, self).__init__()
        self.model: MBartForConditionalGeneration = MBartForConditionalGeneration(mBart_config)  # .to(self.dev)
        # self.model = accelerator.prepare_model(model=self.model)
        #self.accelerator = accelerator
        self.model = nn.DataParallel(self.model, device_ids=device_ids)
        self.dev = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.dev)
        self.scaler = torch.cuda.amp.GradScaler()
        # model.to(device)

    def fit_epoch(self, train_loader, optimizer, epochs, lr_scheduler=None):
        self.model.train(True)
        epoch_loss = 0

        for epoch in range(epochs):
            print("Epoch " + str(epoch))
            epoch_loss = 0
            eta = tqdm(train_loader)
            n_batch = 0
            for batch in eta:
                loss = self.training_step(batch, optimizer, lr_scheduler).item()
                epoch_loss += loss
                eta.set_postfix(loss=str(epoch_loss / (n_batch + 1))[0:6])
                n_batch = n_batch + 1
            torch.save(self.state_dict(), "mbart_v2_test_1k.pt")
            print("Saving weights at epoch " + str(epoch))

    def fit(self, train_loader, optimizer, steps, lr_scheduler=None):
        self.model.train(True)
        epoch_loss = 0
        # eta = tqdm(train_loader)
        eta = tqdm(range(steps))
        # with self.accelerator.accumulate(self):
        train_loader = iter(train_loader)
        for n_batch in eta:
            batch = next(train_loader)

            loss = self.training_step(batch, optimizer, lr_scheduler).item()
            # epoch_loss += loss
            # eta.set_postfix(loss=str(epoch_loss / (n_batch + 1))[0:6])
            eta.set_postfix(loss=str(loss)[0:6])
            if (n_batch + 1) % 1000 == 0:
                torch.save(self.state_dict(), "mbart_v2.pt")
                print("Saving weights at step " + str(n_batch))
            if steps == n_batch: break

    def training_step(self, batch: List[torch.Tensor], optimizer: Optimizer, lr_scheduler) -> Tensor:
        # label_ids, att_mask = batch[0]

        label_ids, att_mask, masked_ids = batch
        label_ids: Tensor = label_ids.to(self.dev)
        att_mask: Tensor = att_mask.to(self.dev)
        masked_ids: Tensor = masked_ids.to(self.dev)

        optimizer.zero_grad(set_to_none=True)
        #with torch.cuda.amp.autocast(device_type='cuda', dtype=torch.float16):
        with torch.cuda.amp.autocast():
            loss = self.model(masked_ids, att_mask, labels=label_ids).loss

        loss = loss.mean()
        #loss.backward()

        self.scaler.scale(loss).backward()

        # Unscales gradients and calls
        # or skips optimizer.step()
        self.scaler.step(optimizer)

        # Updates the scale for next iteration
        self.scaler.update()

        #optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        return loss

    def __call__(self, arg):
        return self.model(arg)

    def evaluate(self, batch: List[torch.Tensor], mask_id: int = 250026):
        input_ids, att_mask, masked_ids = batch
        # input_ids: Tensor = input_ids.to(self.dev)
        # att_mask: Tensor = att_mask.to(self.dev)
        # masked_ids: Tensor = masked_ids.to(self.dev)
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
