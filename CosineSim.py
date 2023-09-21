from typing import List, Dict, Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBartConfig, MBartForConditionalGeneration, \
    MT5ForConditionalGeneration, MT5Config

from utilities.models import ClearCache, get_all_mt6_models, get_all_mbart_models


class CosineSim:

    def __init__(self, model1, model2, device: str = None):
        self.model1 = model1
        self.model2 = model2
        self.model1.eval()
        self.model2.eval()
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def compute_enc_hidd_states(self, dataloader: DataLoader, show_tqdm:bool=True):
        self.model1, self.model2 = self.model1.to(self.device), self.model2.to(self.device)
        avg_cos = 0
        dl = tqdm(dataloader) if show_tqdm else dataloader
        for batch in dl:
            with torch.no_grad():
                with ClearCache():
                    for e in batch:
                        if isinstance(batch[e], torch.Tensor):
                            batch[e] = batch[e].to(self.device)

                    seq2seq_out_m1 = self.model1(**batch)
                    seq2seq_out_m2 = self.model2(**batch)

                    last_hidd_state_m1 = seq2seq_out_m1['encoder_last_hidden_state'].squeeze().T
                    last_hidd_state_m2 = seq2seq_out_m2['encoder_last_hidden_state'].squeeze().T

                    cos_sim_hs_m1_m2 = torch.nn.functional.cosine_similarity(last_hidd_state_m1, last_hidd_state_m2,
                                                                             dim=0)
                    attn_mask = batch["attention_mask"]
                    lens = attn_mask.sum(-1)
                    zero_out_pad_cos_sim = cos_sim_hs_m1_m2 * attn_mask.T
                    avg_cos_sim_per_sent = zero_out_pad_cos_sim.sum(0) / lens
                    # avg_cos_sim_per_sent = torch.mean(cos_sim_hs_m1_m2, dim=0)
                    avg_cos = avg_cos + torch.mean(avg_cos_sim_per_sent) / len(dataloader)
        return float(avg_cos.cpu())

    def compute_logits(self, dataloader: DataLoader, pad_token_id: int):
        self.model1, self.model2 = self.model1.to(self.device), self.model2.to(self.device)
        avg_cos = 0
        for batch in tqdm(dataloader):
            with torch.no_grad():
                with ClearCache():
                    for e in batch:
                        if isinstance(batch[e], torch.Tensor):
                            batch[e] = batch[e].to(self.device)

                    seq2seq_out_m1 = self.model1(**batch)
                    seq2seq_out_m2 = self.model2(**batch)

                    logits_m1 = seq2seq_out_m1['logits'].squeeze().T
                    logits_m2 = seq2seq_out_m2['logits'].squeeze().T

                    cos_sim_hs_m1_m2 = torch.nn.functional.cosine_similarity(logits_m1, logits_m2, dim=0)
                    attn_mask = (batch["labels"] != pad_token_id).long()
                    lens = attn_mask.sum(-1)
                    zero_out_pad_cos_sim = cos_sim_hs_m1_m2 * attn_mask.T
                    avg_cos_sim_per_sent = zero_out_pad_cos_sim.sum(0) / lens
                    # avg_cos_sim_per_sent = torch.mean(cos_sim_hs_m1_m2, dim=0)
                    avg_cos = avg_cos + torch.mean(avg_cos_sim_per_sent) / len(dataloader)
        return float(avg_cos.cpu())


from torch.utils.data import DataLoader
from datasets import Dataset
from typing import Dict, List, Any
from datasets import load_dataset
from transformers import AutoTokenizer


def create_dataloader(trans_pair_ds: Dataset, input_column: str, fn_kwargs: Dict[str, Any],
                      batch_size: int) -> DataLoader:
    trans_pair_ds = trans_pair_ds.map(tokenize, batched=True, input_columns=[input_column],
                                      fn_kwargs=fn_kwargs)
    # trans_pair_ds = trans_pair_ds.remove_columns(column_names=['translation', 'original_text'])
    trans_pair_ds = trans_pair_ds.with_format('torch', columns=["input_ids", "labels", "attention_mask"],
                                              output_all_columns=False)

    # ids = [e['input_ids'].view(1, -1) for e in iter(trans_pair_ds)]
    test_loader = DataLoader(trans_pair_ds, batch_size=batch_size, drop_last=True, pin_memory=False)
    return test_loader


def get_wmt_dataset(lang_pair: str, num_of_rows: int = None) -> Dataset:
    wmt14 = "wmt14"
    split = "test"
    lang_config = lang_pair.split("-")
    assert len(lang_config) == 2
    if lang_config[0] == "en":
        lang_config[0], lang_config[1] = lang_config[1], lang_config[0]
    lang_config = "-".join(lang_config)
    if "es" in lang_config:
        wmt14 = "nikodallanoce/wmt14"
        split = "validation"
    split = split if num_of_rows is None else split + f"[:{num_of_rows}]"
    ds = load_dataset(wmt14, lang_config,
                      cache_dir="/data/n.dallanoce/wmt14",
                      split=split,
                      verification_mode='no_checks')
    return ds


def tokenize(examples: List[Dict[str, str]], **kwargs):
    tokenizer = kwargs['tokenizer']
    src_lang: str = kwargs['lang1']
    tgt_lang: str = kwargs['lang2']
    if "task" in kwargs:
        task: str = kwargs['task']
        batch_src: List[str] = [task + e[src_lang] for e in examples]
    else:
        batch_src: List[str] = [e[src_lang] for e in examples]
    batch_tgt: List[str] = [e[tgt_lang] for e in examples]
    # tokenize the batch of sentences
    outputs = tokenizer(batch_src, text_target=batch_tgt, return_special_tokens_mask=False,
                        add_special_tokens=True, truncation=True,
                        max_length=128, padding='max_length',
                        return_attention_mask=True, return_tensors='pt')
    # labels = tokenizer(batch_tgt, truncation=False)
    # batch_tgt = tokenizer.batch_decode(labels['input_ids'], skip_special_tokens=True)

    return {'input_ids': outputs['input_ids'], 'labels': outputs['labels'], 'attention_mask': outputs['attention_mask']}


# import os
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    from transformers import MBartConfig, MBartForConditionalGeneration
    from utilities.models import get_all_mbart_models

    tok_mbart = AutoTokenizer.from_pretrained("nikodallanoce/mbart-cc4-vanilla-32k-5", src_lang="en_XX",
                                              tgt_lang="fr_XX")

    fn_kwargs_mbart = {'tokenizer': tok_mbart, 'lang1': "en", 'lang2': "fr"}
    wmt14_ds_mbart = get_wmt_dataset(fn_kwargs_mbart['lang1'] + "-" + fn_kwargs_mbart['lang2'], num_of_rows=512)
    dataloader_mbart = create_dataloader(wmt14_ds_mbart, "translation", fn_kwargs_mbart, 32)
    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                               encoder_attention_heads=8, decoder_attention_heads=8,
                               d_model=512, max_length=128, vocab_size=tok_mbart.vocab_size, dropout=0.1)
    rnd_mbart = MBartForConditionalGeneration(mbart_config)

    #mbart_models = get_all_mbart_models()

    cs = CosineSim(rnd_mbart, rnd_mbart)
    sim = cs.compute_enc_hidd_states(dataloader_mbart)
    print(round(sim, 4))
