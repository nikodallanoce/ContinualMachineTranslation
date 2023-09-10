from functools import partial
from typing import Union, Optional, Callable, Dict, List, Tuple, Any

import datasets
import torch
from datasets import load_dataset
from torch import nn, Tensor
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, IterableDataset, BatchSampler
from transformers import Trainer, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback, MT5Tokenizer
from transformers.utils import is_torch_fx_proxy

from custom_datasets.MT6PreTrainingDataset import MT6PreTrainingDataset
from eval.bleu_utility import compute_bleu_mt6
from trainers.CustomTrainer import CustomTrainer
from utilities.utility import collate_pad, collate_torch_iterable, collate_pad_mt6, TrainingStrategy


class MT6Trainer(CustomTrainer):

    def __init__(self, task: TrainingStrategy, model: Union[PreTrainedModel, nn.Module] = None,
                 args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None, train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Callable[[], PreTrainedModel] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
                 tokenizer_name: str = None,
                 batch_sampler: BatchSampler = None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics, tokenizer_name, batch_sampler)
        self.labels = ["labels_pnat", "labels_tsc"] if task == TrainingStrategy.PRE_TRAINING else ["labels"]
        self.task = task

    def compute_bleu(self, eval_ds: Dataset, model: PreTrainedModel, src_lang: str, tgt_lang: str, bleu_type: str,
                     batch_size: int, tokenizer_name: str) -> Dict[str, Any]:
        use_lang_tokens = self.task == TrainingStrategy.FINE_TUNING_LANG
        return compute_bleu_mt6(trans_pair_ds=eval_ds, model=model, src_lang=src_lang, tgt_lang=tgt_lang,
                                bleu_type=bleu_type, batch_size=batch_size, tokenizer_name=tokenizer_name,
                                use_lang_tokens=use_lang_tokens)

    def get_train_dataloader(self) -> DataLoader:

        data_loader: DataLoader

        if type(self.train_dataset) == datasets.iterable_dataset.IterableDataset or isinstance(self.train_dataset,
                                                                                               IterableDataset):
            data_loader = DataLoader(self.train_dataset,
                                     collate_fn=partial(collate_pad_mt6, labels=self.labels,
                                                        pad_token_id=self.model.config.pad_token_id),
                                     batch_size=self.args.per_device_train_batch_size,
                                     pin_memory=self.args.dataloader_pin_memory)

        elif self.batch_sampler is None:
            data_loader = DataLoader(self.train_dataset,
                                     collate_fn=partial(collate_pad_mt6, labels=self.labels,
                                                        pad_token_id=self.model.config.pad_token_id),
                                     batch_size=self.args.per_device_train_batch_size,
                                     drop_last=self.args.dataloader_drop_last,
                                     num_workers=self.args.dataloader_num_workers,
                                     pin_memory=self.args.dataloader_pin_memory,
                                     shuffle=True)
        else:
            data_loader = DataLoader(self.train_dataset,
                                     collate_fn=partial(collate_pad_mt6, labels=self.labels,
                                                        pad_token_id=self.model.config.pad_token_id),
                                     num_workers=self.args.dataloader_num_workers,
                                     pin_memory=self.args.dataloader_pin_memory,
                                     batch_sampler=self.batch_sampler)

        return data_loader

    def compute_loss_mt5(self, model, inputs, return_outputs=False):
        if "labels" in inputs and self.label_smoother is not None:
            inputs['decoder_input_ids'] = self.shift_right(inputs['labels'])
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

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.task == TrainingStrategy.PRE_TRAINING:
            loss = self.compute_mt6_loss(model, inputs, return_outputs)
            return loss
        else:
            inp = inputs[0]
            if self.task == TrainingStrategy.FINE_TUNING_LANG:
                inp["decoder_input_ids"] = self.shift_tokens_right(input_ids=inp["labels"],
                                                                   pad_token_id=self.model.config.pad_token_id)
            return super().compute_loss(model, inp, return_outputs)

    def compute_mt6_loss(self, model, inputs, return_outputs):
        inputs: List[Dict[str, Tensor]]
        d: Dict[str, Tensor]
        inp_dict: Dict[str, Tensor] = {}
        to_pad_tensors: Dict[str, List[Tensor]] = {'input_ids': [], 'attention_mask': [], 'labels': []}
        max_tensor_len = {'input_ids': 0, 'attention_mask': 0, 'labels': 0}
        for d in inputs:
            d_label = list(d.keys())[1]
            to_pad_tensors['input_ids'].append(d['input_ids'])
            max_tensor_len['input_ids'] = d['input_ids'].shape[1] if d['input_ids'].shape[1] > max_tensor_len[
                'input_ids'] else max_tensor_len['input_ids']
            to_pad_tensors['attention_mask'].append(d['attention_mask'])
            max_tensor_len['attention_mask'] = d['attention_mask'].shape[1] if d['attention_mask'].shape[1] > \
                                                                               max_tensor_len['attention_mask'] else \
                max_tensor_len['attention_mask']

            to_pad_tensors['labels'].append(d[d_label])
            max_tensor_len['labels'] = d[d_label].shape[2] if d[d_label].shape[2] > max_tensor_len[
                'labels'] else max_tensor_len['labels']

        for key, tens in to_pad_tensors.items():
            max_len = max_tensor_len[key]
            pad_val = self.model.config.pad_token_id if key != "labels" else -100
            for i in range(len(tens)):
                t = tens[i]
                tens[i] = pad(t, pad=(0, max_len - t.shape[-1]), mode="constant", value=pad_val)
            inp_dict[key] = torch.cat(tens, dim=0)

        loss = super().compute_loss(model, inp_dict, return_outputs)
        return loss
    # def compute_mt6_loss(self, model, inputs, return_outputs):
    #     total_loss: List[torch.Tensor] = []
    #     labels_names: List[str] = []
    #     inputs: List[Dict[str, Tensor]]
    #     for inp in inputs:
    #         label_key = list(inp.keys())[1]
    #         labels_names.append(label_key)
    #         dict_inp: Dict[str, torch.Tensor] = {'input_ids': inp['input_ids'], 'attention_mask': inp['attention_mask'],
    #                                              'labels': inp[label_key]}
    #         # loss: torch.Tensor = super().compute_loss(model, dict_inp, return_outputs)
    #         outputs = model(**dict_inp)
    #         loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    #         total_loss.append(loss)
    #     stack_losses: torch.Tensor = torch.stack(total_loss)
    #     loss: torch.Tensor = torch.sum(stack_losses, dim=0)
    #     if self.state.global_step % self.args.logging_steps == 0 and self.state.global_step > 0:
    #         with torch.no_grad():
    #             if stack_losses.ndim > 1:
    #                 mean_losses: torch.Tensor = torch.mean(stack_losses, dim=1)
    #             else:
    #                 mean_losses: torch.Tensor = stack_losses
    #             for i, lab_name in enumerate(labels_names):
    #                 self.log({lab_name.replace("labels", "loss"): float(mean_losses[i])})
    #     return loss
