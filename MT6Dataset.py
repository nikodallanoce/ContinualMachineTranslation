import math
from random import random
from typing import Any

import datasets
import torch as pt
import torch.utils.data
from datasets.formatting.dataset_wrappers.torch_iterable_dataset import TorchIterableDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MT5Tokenizer, AutoTokenizer, DataCollatorForSeq2Seq, T5Tokenizer, MT5ForConditionalGeneration, \
    T5ForConditionalGeneration
from typing import Tuple, List


class MT6Dataset(torch.utils.data.IterableDataset):

    def __init__(self, tokenized_ds: TorchIterableDataset, tokenizer: T5Tokenizer):
        self.tokenized_ds: TorchIterableDataset = tokenized_ds
        self.tokenizer = tokenizer  # T5Tokenizer.from_pretrained("t5-base")
        self.special_tokens = set(self.tokenizer.all_special_ids)

    @classmethod
    def tokenize_dataset(cls, dataset, input_feat: list[str], fn_kwargs: dict[str, Any]):
        return dataset.map(cls.__tokenize_fn, batched=True, input_columns=input_feat,
                           fn_kwargs=fn_kwargs)

    def mask_dataset(self, tokenized_ids: torch.Tensor, noise_density: float = 0.5) -> Tuple[
        torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        # replace some token that differ from the special ones with a mask token
        if len(tokenized_ids.shape) == 1:
            tokenized_ids = tokenized_ids.view((1, tokenized_ids.shape[0]))
        inp_ids = torch.zeros_like(tokenized_ids)
        labels = torch.zeros_like(tokenized_ids)
        indexes: List[Tuple[int, int]] = []
        for i in range(tokenized_ids.shape[0]):
            mask_ids = iter(self.tokenizer.additional_special_tokens_ids)
            mask_token = next(mask_ids)
            masking_input = False
            masking_lab = False
            k: int = 0
            l: int = 0
            mask_id_usage = 0
            for j in range(0, tokenized_ids.shape[1]):
                # test = tokenized_ids[i][j] in self.special_tokens
                if int(tokenized_ids[i][j]) not in self.special_tokens:
                    if random() <= noise_density:
                        masking_lab = False
                        labels[i][k] = tokenized_ids[i][j]
                        k += 1
                        if not masking_input:
                            inp_ids[i][l] = mask_token
                            l += 1
                            mask_id_usage += 1
                        masking_input = True
                    else:
                        masking_input = False
                        inp_ids[i][l] = tokenized_ids[i][j]
                        l += 1
                        if not masking_lab:
                            labels[i][k] = mask_token
                            indexes.append((i, k))
                            k += 1
                            mask_id_usage += 1
                        masking_lab = True
                    if mask_id_usage == 2:
                        mask_token = next(mask_ids, mask_token)
                        mask_id_usage = 0
                elif int(tokenized_ids[i][j]) == 0:
                    break
                else:
                    inp_ids[i][l] = tokenized_ids[i][j]
                    l += 1
                    labels[i][k] = tokenized_ids[i][j]
                    k += 1

        return inp_ids, labels, indexes

    @staticmethod
    def __tokenize_fn(examples: list, **kwargs):
        tokenizer = kwargs['tokenizer']
        lang = kwargs['lang']
        to_translate = [ex[lang] for ex in examples]
        tokenized_inputs = tokenizer(to_translate, truncation=True, padding='max_length', max_length=16,
                                     return_tensors='pt')
        return tokenized_inputs

    def __iter__(self):
        label = self.tokenized_ds[3]
        label = label['input_ids']
        inp_ids, label, indexes = self.mask_dataset(torch.clone(label), 0.5)
        yield inp_ids, label, indexes


if __name__ == '__main__':
    ccmatrix = datasets.load_dataset("yhavinga/ccmatrix", "en-it", split='train', streaming=True)
    ccmatrix = ccmatrix.remove_columns(['id', 'score'])
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    tokenized_ds = MT6Dataset.tokenize_dataset(ccmatrix, ['translation'], {'lang': 'en', 'tokenizer': tokenizer})
    tokenized_ds: TorchIterableDataset = tokenized_ds.with_format('torch')
    mt6ds = MT6Dataset(tokenized_ds, tokenizer)
    # mt6ds.mask_dataset(tokenizer(["The cute dog walks in the park"], max_length=10, padding="max_length",
    #                             return_tensors='pt').input_ids)

    dataloader = DataLoader(tokenized_ds, batch_size=8)
    mt5: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained('t5-small').to('cuda')
    for i, batch in enumerate(tqdm(dataloader, total=5)):
        inp, lab, indexes = mt6ds.mask_dataset(batch['input_ids'])
        inp = inp.to('cuda')
        # lab: pt.Tensor = lab.view(-1, lab.size(dim=1)).to('cuda')
        lab: pt.Tensor = lab.to('cuda')
        # ris = mt5.forward(inp.to('cuda'), labels=lab)
        # logits = ris.logits
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
        start_ind = 0
        row_ind = 0
        losses = []
        num_of_groups = 2
        span_per_group = 2  # math.ceil(len(indexes) / num_of_groups)
        group_idx = 1
        for j in range(len(indexes)):
            ind = indexes[j]
            end_ind = ind[1]
            if row_ind != ind[0] or j == len(indexes) - 1:
                group_lab = lab[row_ind, start_ind:]
                ris = mt5.forward(inp[row_ind, :].view(1, -1), labels=group_lab.view(1, -1))
                losses.append(ris.loss)
                loss = sum(losses)
                #loss.backward()
                losses = []
                group_idx = 1
                start_ind = 0
                row_ind = ind[0]
            if end_ind == start_ind:
                continue
            if group_idx == span_per_group:
                row_ind = ind[0]
                group_lab = lab[row_ind, start_ind:end_ind + 1]
                start_ind = end_ind
                ris = mt5.forward(inp[row_ind, :].view(1, -1), labels=group_lab.view(1, -1))
                losses.append(ris.loss)
                group_idx = 1
            else:
                group_idx += 1

            # start_ind = ind[1] + 1
        # lls = logits.size(-1)
        # lm_logits = logits.view(-1, lls)
        # v = lab.view(-1)
        tokenized_ds.set_epoch(i)
        if i == 5:
            break
    print()
