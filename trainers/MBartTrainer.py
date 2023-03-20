from functools import partial
from typing import Union, Optional, Callable, Dict, List, Tuple, Any
from utilities.utility import collate_pad
import torch
from datasets import load_dataset
from torch import nn, Tensor
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback, Seq2SeqTrainer, MBartTokenizer

from custom_datasets.MBartTranslationDataset import MBartTranslationDataset


class MBartTrainer(Seq2SeqTrainer):

    def __init__(self, model: Union[PreTrainedModel, nn.Module] = None, args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None, train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None, eval_tokenizers: Dict[str, MBartTokenizer] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Callable[[], PreTrainedModel] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.eval_tokenizers: Dict[str, MBartTokenizer] = eval_tokenizers

    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval", **gen_kwargs) -> Dict[str, float]:

        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          collate_fn=partial(collate_pad, pad_token_id=self.model.config.pad_token_id),
                          batch_size=self.args.per_device_train_batch_size,
                          drop_last=self.args.dataloader_drop_last,
                          num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory,
                          shuffle=True)

    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            inputs['decoder_input_ids'] = self.shift_tokens_right(inputs['labels'], self.model.config.pad_token_id)
        return super().compute_loss(model, inputs, return_outputs)

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
        num_of_cols = prev_output_tokens.shape[-1]
        for row, col_idx in enumerate(index_of_eos):
            del_idx = int(col_idx) + 1
            if del_idx < num_of_cols:
                prev_output_tokens[row, del_idx] = pad_token_id

        return prev_output_tokens
