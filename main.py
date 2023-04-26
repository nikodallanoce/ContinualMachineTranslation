import torch
from datasets import load_dataset, concatenate_datasets, interleave_datasets
from datasets.combine import DatasetType
from torch.utils.data import DataLoader, Dataset, IterableDataset, ConcatDataset, RandomSampler
from tqdm import tqdm
import os

from iterable_datasets.IterMT6PreTrainingDataset import IterMT6PreTrainingDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import MBartTokenizerFast, MT5TokenizerFast
from functools import partial
from utilities.utility import collate_pad, collate_torch_iterable
import custom_datasets.MBartPreTrainingDataset
import custom_datasets.MT6PreTrainingDataset


def map_tokenize(batch, tokenizer: MBartTokenizerFast):
    out_tok = tokenizer(batch, max_length=128, truncation=True, padding="longest")
    return {'input_ids': out_tok['input_ids'], 'attention_mask': out_tok['attention_mask'],
            'labels': out_tok['input_ids']}


if __name__ == '__main__':
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    input = torch.randn(3, 5)
    target = torch.empty(3, dtype=torch.long).fill_(-100)
    output = loss_fn(input, target)

    # tok_name = "nikodallanoce/mbart-cc4-vanilla-32k-5"
    # tok_en = MBartTokenizerFast.from_pretrained(tok_name, src_lang="en_XX")
    # tok_fr = MBartTokenizerFast.from_pretrained(tok_name, src_lang="fr_XX")

    tok_en = MT5TokenizerFast.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5")
    tok_fr = tok_en

    en_mc4 = load_dataset("mc4", "en", split="train", streaming=True)
    fr_mc4 = load_dataset("mc4", "fr", split="train", streaming=True)
    en_mc4 = en_mc4.map(
        partial(custom_datasets.MT6PreTrainingDataset.get_item_for_iterable, tokenizer=tok_en), batched=True,
        batch_size=128,
        remove_columns=['url', 'timestamp'],
        input_columns=['text']).remove_columns(['text'])
    fr_mc4 = fr_mc4.map(
        partial(custom_datasets.MT6PreTrainingDataset.get_item_for_iterable, tokenizer=tok_fr), batched=True,
        batch_size=128,
        remove_columns=['url', 'timestamp'],
        input_columns=['text'])
    pre_ds = interleave_datasets([en_mc4, fr_mc4]).with_format("torch")
    # ConcatDataset([en_mc4, fr_mc4])
    en_fr_ds = load_dataset("yhavinga/ccmatrix", "en-fr",
                            split="train",
                            streaming=True)
    en_fr_ds = en_fr_ds.map(
        partial(custom_datasets.MT6PreTrainingDataset.get_item_for_iterable, tokenizer=tok_en,
                has_translation_pairs=True), batched=True,
        batch_size=128,
        remove_columns=['id', 'score'],
        input_columns=['translation']).remove_columns(['translation'])

    n1 = next(iter(en_fr_ds))
    n2 = next(iter(en_mc4))
    #pre_train_ds = interleave_datasets([en_mc4, en_fr_ds]).with_format("torch")
    for e in tqdm(
            DataLoader(IterMT6PreTrainingDataset([en_mc4, en_fr_ds]), batch_size=4,
                       collate_fn=partial(collate_torch_iterable, pad_token_id=tok_en.pad_token_id),
                       pin_memory=True)):
        pass
