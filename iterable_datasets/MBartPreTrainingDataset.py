from typing import Iterator
import datasets
import torch
from transformers import MBartTokenizerFast
from noise_functions.MBartNoiseFunction import MBartNoiseFunction


class MBartPreTrainingDataset(torch.utils.data.IterableDataset):

    def __init__(self, hugg_dataset: datasets.IterableDataset, tokenizer: MBartTokenizerFast,
                 input_max_length=128, ds_field: str = "text", seed: int = 0):
        super(MBartPreTrainingDataset, self).__init__()
        # self.ds_indexes: Dict[str, Tuple[int, int]] = prepare_ds_indexes(hugg_datasets)
        self.dataset: Iterator = iter(hugg_dataset)  # datasets.concatenate_datasets(list(hugg_datasets.values()))
        self.tokenizer: MBartTokenizerFast = tokenizer
        self.input_max_length: int = input_max_length
        self.noise_fn = MBartNoiseFunction()
        self.ds_field: str = ds_field
        self.ref_len: float = input_max_length - input_max_length * 0.4
        self.seed: int = seed

    def __iter__(self) -> Iterator:
        field = self.ds_field
        label_ids = next(self.dataset)[field]

        label_ids = label_ids.strip()

        label_ids, masked_ids = self.noise_fn.compute(label_ids, self.seed)
        tokenized = self.tokenizer([label_ids, masked_ids], return_special_tokens_mask=False,
                                   add_special_tokens=True, truncation=True, return_attention_mask=False,
                                   max_length=self.input_max_length, padding='longest',
                                   return_tensors='pt')['input_ids']

        label_ids = tokenized[0].view(-1)
        masked_ids = tokenized[1].view(-1)

        att_mask = torch.where(masked_ids != self.tokenizer.pad_token_id, 1, 0)
        label_ids = torch.where(label_ids == self.tokenizer.pad_token_id, -100, label_ids)
        yield {'input_ids': masked_ids, 'labels': label_ids, 'attention_mask': att_mask}
