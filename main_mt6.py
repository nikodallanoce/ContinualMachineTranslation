import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, T5Config
from tqdm import tqdm
from custom_datasets.MT6PreTrainingDataset import MT6PreTrainingDataset
import torch
from torch import Tensor
from typing import Set, List


def create_groups(targets: torch.Tensor, masking_tokens: Set[int]):
    groups: List[Tensor] = []

    group_i = []
    for i in range(len(targets)):
        tok_id = targets[i]
        tok_id


def compute_loss(model: MT5ForConditionalGeneration, data):
    input_ids = data["input_ids"]
    targets = data["labels"]
    att_mask = data["attention_mask"]
    loss = 0
    for i in range(targets.shape[1]):
        groups: Tensor = targets[:, i, :]
        labels = groups.contiguous()
        loss += model(input_ids=input_ids, attention_mask=att_mask, labels=labels).loss
        print()


if __name__ == '__main__':
    pre_train_ds = load_dataset("text", data_files={"train": ["D:\\datasets\\test_hugg_en\\test_data_hugg.txt"]},
                                cache_dir="D:\\datasets\\test_hugg_en", split='train',
                                ignore_verifications=True)

    # translation_ds = load_dataset("text", data_files={"train": ["/data/n.dallanoce/cc100/en.txt"]},
    #                             cache_dir="/data/n.dallanoce/cc100/hugg_en", split=f"train[:50%]",
    #                             ignore_verifications=True)

    tok_en = MT5Tokenizer.from_pretrained("google/mt5-base")
    model = MT5ForConditionalGeneration(
        T5Config(vocab_size=tok_en.vocab_size, decoder_start_token_id=tok_en.pad_token_id))

    data_l = DataLoader(MT6PreTrainingDataset(pre_train_ds, tok_en), batch_size=2, drop_last=True)

    masking_tokens_ids = tok_en(" ".join([f'<extra_id_{i}>' for i in range(100)]), add_special_tokens=False)[
        'input_ids']
    masking_tokens_ids = set(masking_tokens_ids)

    for e in tqdm(data_l):
        compute_loss(model, e)
