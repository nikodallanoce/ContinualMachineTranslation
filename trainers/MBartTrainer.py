from functools import partial
from typing import Union, Optional, Callable, Dict, List, Tuple, Any

import datasets
import numpy as np
from transformers.trainer_utils import EvalLoopOutput

from utilities.utility import collate_pad, collate_torch_iterable
import torch

from torch import nn, Tensor

from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback, Seq2SeqTrainer, MBartTokenizer
from eval.bleu_utility import compute_bleu_mbart


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

    # def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
    #     return DataLoader(eval_dataset,
    #                       collate_fn=partial(collate_pad, pad_token_id=self.model.config.pad_token_id),
    #                       batch_size=self.args.per_device_eval_batch_size,
    #                       drop_last=self.args.dataloader_drop_last,
    #                       num_workers=self.args.dataloader_num_workers,
    #                       pin_memory=False,
    #                       shuffle=False)

    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval", **gen_kwargs) -> Dict[str, float]:

        eval_metrics: Dict[str, float] = dict()
        src_tgt_langs = metric_key_prefix.split("_")
        src_lang, tgt_lang = src_tgt_langs[1], src_tgt_langs[2]
        for i in range(2):
            bleu_score: Dict[str, Any] = compute_bleu_mbart(eval_dataset, self.model,
                                                            src_lang=src_lang,
                                                            tgt_lang=tgt_lang)
            metric_key = next(iter(bleu_score))
            eval_metrics[f"eval_bleu_{src_lang}_{tgt_lang}"] = bleu_score[metric_key]
            src_lang, tgt_lang = tgt_lang, src_lang

        self.log(eval_metrics)
        return eval_metrics

    # def evaluation_loop(self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None,
    #                     ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> EvalLoopOutput:
    #     eval_metrics: Dict[str, float] = dict()
    #     src_tgt_langs = metric_key_prefix.split("_")
    #     # bleu_score = compute_bleu(eval_dataset, self.model,
    #     #                           lang1=src_tgt_langs[2],
    #     #                           lang2=src_tgt_langs[3])["bleu"] * 100
    #     bleu_score = 32
    #     eval_metrics[metric_key_prefix] = bleu_score
    #
    #     return super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

    def get_train_dataloader(self) -> DataLoader:
        data_loader: DataLoader
        if type(self.train_dataset) == datasets.iterable_dataset.IterableDataset:
            data_loader = DataLoader(self.train_dataset,
                                     collate_fn=partial(collate_torch_iterable,
                                                        pad_token_id=self.model.config.pad_token_id),
                                     batch_size=self.args.per_device_train_batch_size,
                                     pin_memory=self.args.dataloader_pin_memory)
        else:
            data_loader = DataLoader(self.train_dataset,
                                     collate_fn=partial(collate_pad, pad_token_id=self.model.config.pad_token_id),
                                     batch_size=self.args.per_device_train_batch_size,
                                     drop_last=self.args.dataloader_drop_last,
                                     num_workers=self.args.dataloader_num_workers,
                                     pin_memory=self.args.dataloader_pin_memory,
                                     shuffle=True)
        return data_loader

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
