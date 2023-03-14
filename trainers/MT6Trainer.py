from functools import partial
from typing import Union, Optional, Callable, Dict, List, Tuple, Any

import torch
from datasets import load_dataset
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback, Seq2SeqTrainer, MT5Tokenizer
from transformers.utils import is_torch_fx_proxy

from custom_datasets.MT6PreTrainingDataset import MT6PreTrainingDataset
from utilities.utility import collate_pad


class MT6Trainer(Seq2SeqTrainer):

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

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          collate_fn=partial(collate_pad, pad_token_id=self.model.config.pad_token_id),
                          batch_size=self.args.per_device_train_batch_size,
                          drop_last=self.args.dataloader_drop_last,
                          num_workers=self.args.dataloader_num_workers,
                          pin_memory=self.args.dataloader_pin_memory)

    def compute_loss_mt5(self, model, inputs, return_outputs=False):
        if "labels" in inputs and self.label_smoother is not None:
            inputs['decoder_input_ids'] = self.shift_right(inputs['labels'])
        return super().compute_loss(model, inputs, return_outputs)

    def shift_right(self, input_ids):
        decoder_start_token_id = self.model.config.decoder_start_token_id
        pad_token_id = self.model.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     targets = inputs["labels"]
    #     # att_mask = inputs["attention_mask"]
    #     loss: Tensor = 0
    #     for i in range(targets.shape[1]):
    #         groups: Tensor = targets[:, i, :]
    #         labels = groups.contiguous()
    #         inputs["labels"] = labels
    #         loss += super().training_step(model, inputs)
    #     return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        # input_ids = inputs["input_ids"]
        targets = inputs["labels"]
        # att_mask = inputs["attention_mask"]
        loss: Tensor = 0
        for i in range(targets.shape[1]):
            groups: Tensor = targets[:, i, :]
            labels: Tensor = groups.contiguous()
            # decoder_input_ids = self.shift_tokens_right(labels, model.config.pad_token_id)
            # if int(labels[:, 0]) == model.config.eos_token_id:
            #     continue
            inputs["labels"] = labels
            inputs["decoder_input_ids"] = self.shift_right(labels)
            loss += super().compute_loss(model, inputs, return_outputs)
        return loss
