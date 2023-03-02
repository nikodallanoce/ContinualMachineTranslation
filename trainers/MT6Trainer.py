from typing import Union, Optional, Callable, Dict, List, Tuple, Any

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from transformers import Trainer, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback, Seq2SeqTrainer


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
            if int(labels[:, 0]) == model.config.eos_token_id:
                continue
            inputs["labels"] = labels
            loss += super().compute_loss(model, inputs, return_outputs)
        return loss

    @staticmethod
    def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
        """
        Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
        have a single `decoder_start_token_id` in contrast to other Bart-like models.
        """
        prev_output_tokens = input_ids.clone()

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

        index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
        prev_output_tokens[:, 0] = decoder_start_tokens

        return prev_output_tokens
