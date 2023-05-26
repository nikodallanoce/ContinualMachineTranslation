from functools import partial
from typing import Union, Optional, Callable, Dict, List, Tuple, Any

import datasets
import torch
from datasets import load_dataset
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import Trainer, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback, Seq2SeqTrainer, MT5Tokenizer
from transformers.utils import is_torch_fx_proxy

from custom_datasets.MT6PreTrainingDataset import MT6PreTrainingDataset
from eval.bleu_utility import compute_bleu_mt6
from utilities.utility import collate_pad, collate_torch_iterable, collate_pad_mt6


class MT6Trainer(Seq2SeqTrainer):

    def __init__(self, task: str, model: Union[PreTrainedModel, nn.Module] = None, args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None, train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None, tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Callable[[], PreTrainedModel] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None
                 ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.labels = ["labels_pnat", "labels_tsc"] if task == "pretraining" else ["labels"]

    def get_train_dataloader(self) -> DataLoader:

        data_loader: DataLoader

        if type(self.train_dataset) == datasets.iterable_dataset.IterableDataset or isinstance(self.train_dataset,
                                                                                               IterableDataset):
            data_loader = DataLoader(self.train_dataset,
                                     collate_fn=partial(collate_pad_mt6, labels=self.labels,
                                                        pad_token_id=self.model.config.pad_token_id, num_workers=1),
                                     batch_size=self.args.per_device_train_batch_size,
                                     pin_memory=self.args.dataloader_pin_memory)

        else:
            data_loader = DataLoader(self.train_dataset,
                                     collate_fn=partial(collate_pad_mt6, labels=self.labels,
                                                        pad_token_id=self.model.config.pad_token_id),
                                     batch_size=self.args.per_device_train_batch_size,
                                     drop_last=self.args.dataloader_drop_last,
                                     num_workers=self.args.dataloader_num_workers,
                                     pin_memory=self.args.dataloader_pin_memory,
                                     shuffle=True)
        return data_loader

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

    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval", **gen_kwargs) -> Dict[str, float]:

        eval_metrics: Dict[str, float] = dict()
        src_tgt_langs = metric_key_prefix.split("_")
        src_lang, tgt_lang = src_tgt_langs[1], src_tgt_langs[2]
        for i in range(2):
            bleu_score = compute_bleu_mt6(eval_dataset, self.model,
                                          src_lang=src_lang,
                                          tgt_lang=tgt_lang,
                                          batch_size=32)
            metric_key = next(iter(bleu_score))
            eval_metrics[f"eval_bleu_{src_lang}_{tgt_lang}"] = bleu_score[metric_key]
            src_lang, tgt_lang = tgt_lang, src_lang

        self.log(eval_metrics)
        return eval_metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.metric_for_best_model is None or "loss" in self.args.metric_for_best_model:
            loss = self.compute_mt6_loss(model, inputs, return_outputs)
            return loss
        else:
            inp = inputs[0]
            return super().compute_loss(model, inp, return_outputs)

    def compute_mt6_loss(self, model, inputs, return_outputs):
        total_loss: List[torch.Tensor] = []
        labels_names: List[str] = []
        inputs: List[Dict[str, Tensor]]
        for inp in inputs:
            label_key = list(inp.keys())[1]
            labels_names.append(label_key)
            dict_inp: Dict[str, torch.Tensor] = {'input_ids': inp['input_ids'], 'attention_mask': inp['attention_mask'],
                                                 'labels': inp[label_key]}
            # loss: torch.Tensor = super().compute_loss(model, dict_inp, return_outputs)
            outputs = model(**dict_inp)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            total_loss.append(loss)
        stack_losses: torch.Tensor = torch.stack(total_loss)
        loss: torch.Tensor = torch.sum(stack_losses, dim=0)
        if self.state.global_step % self.args.logging_steps == 0 and self.state.global_step > 0:
            with torch.no_grad():
                if stack_losses.ndim > 1:
                    mean_losses: torch.Tensor = torch.mean(stack_losses, dim=1)
                else:
                    mean_losses: torch.Tensor = stack_losses
                for i, lab_name in enumerate(labels_names):
                    self.log({lab_name.replace("labels", "loss"): float(mean_losses[i])})
        return loss
