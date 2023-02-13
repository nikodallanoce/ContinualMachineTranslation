from typing import Union, Optional, Callable, Dict, List, Tuple, Any

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback, Seq2SeqTrainer


class CustomTrainer(Seq2SeqTrainer):

    def __init__(self, model: Union[PreTrainedModel, nn.Module] = None, args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None, train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None, tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Callable[[], PreTrainedModel] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.train_loss = 0
        self.step = 1

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     ris = super().compute_loss(model, inputs, return_outputs)
    #     epoch = self.state.epoch
    #     loss = ris.item()
    #     if round(epoch - int(epoch), 10) > 0:
    #         self.train_loss += loss
    #         self.step += 1
    #     else:
    #         self.train_loss = loss
    #         self.step = 1
    #     if self.state.global_step % 50 == 0:
    #         self.log({"train_loss": self.train_loss / self.step})
    #     return ris

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     ris = super().compute_loss(model, inputs, return_outputs)
    #     loss = ris.item()
    #
    #     if self.state.global_step % 100 != 0:
    #         self.train_loss += loss
    #         self.step += 1
    #     else:
    #         self.log({"train_loss": self.train_loss / self.step})
    #         self.train_loss = loss
    #         self.step = 1
    #
    #     return ris
