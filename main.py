import torch
from datasets import load_dataset, concatenate_datasets, interleave_datasets
from datasets.combine import DatasetType
from torch import Tensor
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset, IterableDataset, ConcatDataset, RandomSampler
from tqdm import tqdm
import os

from MT6TokenizerFast import MT6TokenizerFast
from iterable_datasets.IterMT6PreTrainingDataset import IterMT6PreTrainingDataset
from noise_functions.MT6NoiseFunction import MT6NoiseFunction

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import MBartTokenizerFast, MT5TokenizerFast
from functools import partial
from utilities.utility import collate_pad, collate_torch_iterable, collate_pad_mt6
import custom_datasets.MBartPreTrainingDataset
import custom_datasets.MT6PreTrainingDataset
from typing import List, Union


def map_tokenize(batch, tokenizer: MBartTokenizerFast):
    out_tok = tokenizer(batch, max_length=128, truncation=True, padding="longest")
    return {'input_ids': out_tok['input_ids'], 'attention_mask': out_tok['attention_mask'],
            'labels': out_tok['input_ids']}


def create_groups(labels: torch.Tensor):
    groups = []
    transl = []
    start_i = 0
    for r in range(labels.shape[0]):
        labels_row = labels[r, :]
        groups_row = []
        for i in range(0, len(labels_row) - 1):
            if 3 <= int(labels_row[i]) <= 102 and labels_row[i] == labels_row[i + 1]:
                groups_row.append(labels_row[start_i: i + 1])
                start_i = i + 1
        if start_i < len(labels_row):
            groups_row.append([labels_row[start_i:]])
        if len(groups_row) == 0:
            transl.append(labels_row)
        else:
            max_len_labels = max([x.shape[0] for x in groups_row])
            for i in range(len(groups_row)):
                t: Tensor = groups_row[i]
                groups_row[i] = pad(t, pad=(0, max_len_labels - t.shape[-1]), mode="constant", value=-100)
            groups.append(torch.stack(groups_row))
    return groups, transl


if __name__ == '__main__':
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    input = torch.randn(3, 5)
    target = torch.empty(3, dtype=torch.long).fill_(1)
    target[2] = -100
    output = loss_fn(input, target)
    output_2 = loss_fn(input[0:2, :], target[0:2])

    en_es_wmt = load_dataset("nikodallanoce/wmt14", "es-en", split="test", streaming=True, use_auth_token=True)

    tok_name = "nikodallanoce/mbart-cc4-vanilla-32k-5"
    bart_tok_en = MBartTokenizerFast.from_pretrained(tok_name, src_lang="en_XX")
    bart_tok_en("hello man.")
    # tok_fr = MBartTokenizerFast.from_pretrained(tok_name, src_lang="fr_XX")

    lang_codes = ["en_XX", "fr_XX", "es_XX", "de_DE"]
    additional_special_tokens = [f"<extra_id_{i}>" for i in range(100)] + lang_codes

    tok_en = MT5TokenizerFast.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5",
                                              additional_special_tokens=additional_special_tokens, src_lang="en_XX",
                                              tgt_lang="fr_XX")

    lang_code_to_id = {lang: len(tok_en) - len(lang_codes) + i for i, lang in enumerate(lang_codes)}
    id_to_lang_code = {v: k for k, v in lang_code_to_id.items()}
    tok_en = MT5TokenizerFast.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5",
                                              additional_special_tokens=additional_special_tokens, src_lang="en_XX",
                                              tgt_lang="fr_XX", lang_code_to_id=lang_code_to_id,
                                              id_to_lang_code=id_to_lang_code)

    mt6_tok: MT5TokenizerFast = MT6TokenizerFast.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5",
                                                                 additional_special_tokens=additional_special_tokens,
                                                                 src_lang="en_XX",
                                                                 tgt_lang="fr_XX",
                                                                 lang_code_to_id=lang_code_to_id,
                                                                 id_to_lang_code=id_to_lang_code
                                                                 )

    mt6_tok.push_to_hub("nikodallanoce/mt6_tok_fast")

    mt6_hub = MT6TokenizerFast.from_pretrained("nikodallanoce/mt6_tok_fast", src_lang="en_XX", tgt_lang="es_XX")

    mt6_out= mt6_hub("hello my friend.", text_target="good boy")
    sp_tok = tok_en("<extra_id_0><extra_id_99><extra_id_1>", text_target="<extra_id_0>", add_special_tokens=True)

    tok_fr = tok_en

    en_mc4 = load_dataset("mc4", "en", split="train", streaming=True)
    fr_mc4 = load_dataset("mc4", "fr", split="train", streaming=True)
    en_mc4 = en_mc4.map(
        partial(custom_datasets.MT6PreTrainingDataset.get_item_for_iterable, tokenizer=tok_en,
                noise_fn=MT6NoiseFunction(pnat=True)),
        batched=True,
        batch_size=128,
        remove_columns=['url', 'timestamp'],
        input_columns=['text']).remove_columns(['text'])
    fr_mc4 = fr_mc4.map(
        partial(custom_datasets.MT6PreTrainingDataset.get_item_for_iterable, tokenizer=tok_fr), batched=True,
        batch_size=128,
        remove_columns=['url', 'timestamp'],
        input_columns=['text'])

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
    # pre_train_ds = interleave_datasets([en_mc4, en_fr_ds]).with_format("torch")
    for e in tqdm(
            DataLoader(IterMT6PreTrainingDataset([en_mc4, en_fr_ds]), batch_size=4,
                       collate_fn=partial(collate_pad_mt6, pad_token_id=tok_en.pad_token_id),
                       pin_memory=True)):
        # ris = create_groups(e['labels'])
        pass
