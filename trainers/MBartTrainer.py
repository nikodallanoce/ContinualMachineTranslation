from functools import partial
from typing import Union, Optional, Callable, Dict, List, Tuple, Any
from utilities.utility import collate_pad
import torch
from datasets import load_dataset
from torch import nn, Tensor
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback, Seq2SeqTrainer, MBartTokenizer

from custom_datasets.MBartTranslationDataset import MBartTranslationDataset


class MBartTrainer(Seq2SeqTrainer):

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

    # def collate_pad(self, batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    #     inp_ids_list: List[Tensor] = []
    #     labels_list: List[Tensor] = []
    #     att_mask_list: List[Tensor] = []
    #     elem: Dict[str, Tensor]
    #     for elem in batch:
    #         inp_ids_list.append(elem['input_ids'])
    #         labels_list.append(elem['labels'])
    #         att_mask_list.append(elem['attention_mask'])
    #
    #     padding_id = self.model.config.pad_token_id
    #     padded_inp: Tensor = pad_sequence(inp_ids_list, True, padding_value=padding_id)
    #     padded_lab: Tensor = pad_sequence(labels_list, True, padding_value=-100)
    #     padded_att: Tensor = pad_sequence(att_mask_list, True, padding_value=0)
    #     tgt_len = max([padded_inp.shape[-1], padded_lab.shape[-1], padded_att.shape[-1]])
    #     padded_inp: Tensor = pad(padded_inp, pad=(0, tgt_len - padded_inp.shape[-1], 0, 0), mode='constant',
    #                              value=padding_id)
    #     padded_att: Tensor = pad(padded_att, pad=(0, tgt_len - padded_att.shape[-1], 0, 0), mode='constant', value=0)
    #     padded_lab: Tensor = pad(padded_lab, pad=(0, tgt_len - padded_lab.shape[-1], 0, 0), mode='constant', value=-100)
    #     return {"input_ids": padded_inp, "labels": padded_lab, "attention_mask": padded_att}

    def get_train_dataloader(self) -> DataLoader:
        # dataset = load_dataset("wmt14", "fr-en",
        #                        cache_dir="D:\\datasets\\wmt14",
        #                        split=f"train[0:2048]",
        #                        ignore_verifications=True)

        dataset = load_dataset("yhavinga/ccmatrix", "en-fr",
                               cache_dir="/data/n.dallanoce/cc_en_fr",
                               split=f"train[0:20000000]",
                               ignore_verifications=True)

        tok_en = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="fr_XX")

        dataset = MBartTranslationDataset(dataset, tok_en, "fr", input_max_length=64)

        return DataLoader(dataset, collate_fn=partial(collate_pad, pad_token_id=self.model.config.pad_token_id),
                          batch_size=self.args.per_device_train_batch_size,
                          drop_last=self.args.dataloader_drop_last,
                          num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory)

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
