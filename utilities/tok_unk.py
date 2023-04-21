from typing import List, Dict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, MBartTokenizerFast, MBartTokenizer


def tokenize(examples: List[Dict[str, str]], **kwargs):
    tokenizer: MBartTokenizerFast = kwargs['tokenizer']
    src_lang: str = kwargs['lang1']
    tgt_lang: str = kwargs['lang2']
    batch_src: List[str] = [e[src_lang] for e in examples]
    batch_tgt: List[str] = [e[tgt_lang] for e in examples]
    # tokenize the batch of sentences
    outputs = tokenizer(batch_src, text_target=batch_tgt, return_special_tokens_mask=False,
                        add_special_tokens=True, truncation=True, return_attention_mask=False,
                        max_length=128, padding="max_length",
                        return_tensors='pt')

    return {'input_ids': outputs['input_ids'], 'labels': outputs['labels']}


if __name__ == '__main__':
    src_lang = "en"
    tgt_lang = "de"
    translation_ds = load_dataset("yhavinga/ccmatrix", f"{src_lang}-{tgt_lang}",
                                  cache_dir=f"/data/n.dallanoce/cc_{src_lang}_{tgt_lang}",
                                  split=f"train[0:100%]",
                                  verification_mode='no_checks')

    mbart_tgt = f"{tgt_lang}_XX"
    if tgt_lang == "de":
        mbart_tgt = f"{tgt_lang}_DE"

    tok = MBartTokenizerFast.from_pretrained("/home/n.dallanoce/PyCharm/pretraining/tokenizers/mbart",
                                             src_lang=f"{src_lang}_XX",
                                             tgt_lang=mbart_tgt)

    dataset = translation_ds.map(tokenize, batched=True, input_columns=['translation'], num_proc=128,
                                 fn_kwargs={'tokenizer': tok, 'lang1': src_lang, 'lang2': tgt_lang})
    dataset = dataset.with_format("pytorch", columns=['input_ids', 'labels'])

    num_unk_src = 0
    num_unk_tgt = 0
    for batch in tqdm(DataLoader(dataset, batch_size=1024, drop_last=True, num_workers=32)):
        tok_src: torch.Tensor = batch['input_ids'].to("cuda:0")
        tok_tgt = batch['labels'].to("cuda:0")
        num_unk_src += tok_src.view(-1).eq(tok.unk_token_id).sum()
        num_unk_tgt += tok_tgt.view(-1).eq(tok.unk_token_id).sum()
    print(num_unk_src)
    print(num_unk_tgt)
