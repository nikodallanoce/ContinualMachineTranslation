from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBartForConditionalGeneration, \
    MT5ForConditionalGeneration
import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import paper_cka
import transformers
from datasets import load_dataset
from tqdm import tqdm

transformers.logging.set_verbosity_error()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import inspect, re
from typing import List, Union, Dict


def varname(p) -> str:
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return str(m.group(1))


import gc


class ClearCache:

    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        torch.cuda.empty_cache()


from typing import List, Dict, Any

PREFIX_TASK = {'en': "English", 'fr': "French", 'de': "German", 'es': "Spanish"}


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
                        return_attention_mask=False, return_tensors='pt')
    # labels = tokenizer(batch_tgt, truncation=False)
    # batch_tgt = tokenizer.batch_decode(labels['input_ids'], skip_special_tokens=True)

    return {'input_ids': outputs['input_ids'], 'labels': outputs['labels']}


from torch.utils.data import DataLoader
from datasets import Dataset


def create_dataloader(trans_pair_ds: Dataset, input_column: str, fn_kwargs: Dict[str, Any],
                      batch_size: int) -> DataLoader:
    trans_pair_ds = trans_pair_ds.map(tokenize, batched=True, input_columns=[input_column],
                                      fn_kwargs=fn_kwargs)
    # trans_pair_ds = trans_pair_ds.remove_columns(column_names=['translation', 'original_text'])
    trans_pair_ds = trans_pair_ds.with_format('torch', columns=["input_ids", "labels"], output_all_columns=False)

    # ids = [e['input_ids'].view(1, -1) for e in iter(trans_pair_ds)]
    test_loader = DataLoader(trans_pair_ds, num_workers=2, batch_size=batch_size, drop_last=True, pin_memory=False)
    return test_loader


def get_comparison_set(src_lang: str, tgt_lang: str, tok_model: str, split: str = "validation[0:128]",
                       batch_size: int = 8):
    fn_kwargs = {'lang1': src_lang, 'lang2': tgt_lang}

    lang_2 = fn_kwargs['lang1'] if fn_kwargs['lang1'] != "en" else fn_kwargs['lang2']
    val_set_name = "nikodallanoce/wmt14" if src_lang == "es" or tgt_lang == "es" else "wmt14"
    wmt14_val = load_dataset(val_set_name, f"{lang_2}-en",
                             cache_dir="/data/n.dallanoce/wmt14",
                             split=split,
                             verification_mode='no_checks')
    tok = None
    if tok_model == "mbart":
        src_mbart = fn_kwargs['lang1'] + "_XX" if fn_kwargs['lang1'] != "de" else fn_kwargs['lang1'] + "_DE"
        tgt_mbart = fn_kwargs['lang2'] + "_XX" if fn_kwargs['lang2'] != "de" else fn_kwargs['lang2'] + "_DE"
        tok = AutoTokenizer.from_pretrained("nikodallanoce/mbart-cc4-vanilla-32k-5", src_lang=src_mbart,
                                            tgt_lang=tgt_mbart)
    elif tok_model == "mt6":
        tok = AutoTokenizer.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5")
    assert tok is not None
    fn_kwargs['tokenizer'] = tok

    return create_dataloader(wmt14_val, "translation", fn_kwargs, batch_size)


if __name__ == '__main__':

    checkpoints_dir = "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_ft_en-fr-Mf1_weights_anlsys/"
    directories = os.listdir(checkpoints_dir)
    list_models: Dict[str, Union[MBartForConditionalGeneration, MT5ForConditionalGeneration]] = {}
    print(f"Loading these models: {directories}")
    for i, chckpnt in enumerate(tqdm(directories)):
        if chckpnt.startswith("checkpoint"):
            model_dir = os.path.join(checkpoints_dir, chckpnt)
            list_models[chckpnt] = AutoModelForSeq2SeqLM.from_pretrained(os.path.abspath(model_dir),
                                                                         output_attentions=True)
    step_ft_models = list(list_models.values())

    from transformers import MBartConfig

    mbart_tok = AutoTokenizer.from_pretrained("nikodallanoce/mbart-cc4-vanilla-32k-5", src_lang="en_XX",
                                              tgt_lang="fr_XX")
    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                               encoder_attention_heads=8, decoder_attention_heads=8,
                               d_model=512, max_length=128, vocab_size=mbart_tok.vocab_size, dropout=0.1)
    random_model: MBartForConditionalGeneration = MBartForConditionalGeneration(mbart_config)

    only_ft_en_fr: Union[
        MBartForConditionalGeneration, MT5ForConditionalGeneration] = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_ft_en-fr-Mf1_ft_only/checkpoint-100000",
        output_attentions=True)

    last_ft_model = list(list_models.values())[-1]

    m1_model = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_pre_en-fr(M1)/checkpoint-180000",
        output_attentions=True)
    m2_model = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_pre_en-fr_de(M2)/checkpoint-180000",
        output_attentions=True)
    m2_model_rply = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_pre_en-fr_de(M2)_replay_8/checkpoint-180000",
        output_attentions=True)
    m3_model = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_pre_en-fr_de_es(M3)/checkpoint-180000",
        output_attentions=True)
    m3_model_rply = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_pre_en-fr_de_es(M3)_replay_8/checkpoint-180000",
        output_attentions=True)
    de_model: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_pre_en-de/checkpoint-180000", output_attentions=True)

    # mf1_2_rply_model: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
    #     "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_pre_de_ft_en-fr(Mf1-2)_replay_8/checkpoint-100000",
    #     output_attentions=True)

    # mf1_2_model: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
    #     "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_pre_de_ft_en-fr(Mf1-2)/checkpoint-100000",
    #     output_attentions=True)

    mf1_model: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_ft_en-fr-Mf1_weights_anlsys/checkpoint-100000",
        output_attentions=True)

    mf2_model: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_ft_en-de-Mf2/checkpoint-100000",
        output_attentions=True)

    mf2_model_rply: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_ft_en-de-Mf2_replay_8/checkpoint-100000",
        output_attentions=True)

    mf23_model: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_pre_es_ft_en-de(Mf2-3)/checkpoint-80000",
        output_attentions=True)
    mf23_model_rply: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_pre_es_ft_en-de(Mf2-3)_replay_8/checkpoint-95000",
        output_attentions=True)
    mf13_model: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_pre_es_ft_en-fr(Mf3-1)/checkpoint-85000",
        output_attentions=True)
    mf13_model_rply: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_pre_es_ft_en-fr(Mf3-1)_replay_8/checkpoint-100000",
        output_attentions=True)

    only_ft_en_de: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_ft_en-de_ft_only/checkpoint-100000",
        output_attentions=True)

    mf3_model: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_ft_en-es/checkpoint-100000",
        output_attentions=True)

    mf3_model_rply: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_ft_en-es_replay_8/checkpoint-100000",
        output_attentions=True)

    mbart_str2model = {
        "M1": m1_model,
        "M2": m2_model,
        "M2_replay": m2_model_rply,
        "M3": m3_model,
        "M3_replay": m3_model_rply,
        "MF1": mf1_model,
        "MF2": mf2_model,
        "MF2_replay": mf2_model_rply,
        "MF3": mf3_model,
        "MF3_replay": mf3_model_rply,
        "MF2-3": mf23_model,
        "MF2-3_replay": mf23_model_rply,
        "MF1-3": mf13_model,
        "MF1-3_replay": mf13_model_rply
    }

    models_to_comp = [(k, v) for k, v in mbart_str2model.items()]

    layers_enc_norm = [f"model.encoder.layers.{i}.self_attn_layer_norm" for i in range(6)] + [
        f"model.encoder.layers.{i}.final_layer_norm" for i in range(6)]
    layers_enc_fc = [f"model.encoder.layers.{i}.fc1" for i in range(6)] + [f"model.encoder.layers.{i}.fc2" for i in
                                                                           range(6)]
    layers_dec_fc = [f"model.decoder.layers.{i}.fc1" for i in range(6)] + [f"model.decoder.layers.{i}.fc2" for i in
                                                                           range(6)]
    layers_dec_norm = [f"model.decoder.layers.{i}.self_attn_layer_norm" for i in range(6)] + [
        f"model.decoder.layers.{i}.final_layer_norm" for i in range(6)] + [
                          f"model.decoder.layers.{i}.encoder_attn_layer_norm" for i in range(6)]

    lang_pairs = [("en", "fr"), ("en", "de"), ("en", "es"), ("fr", "en"), ("de", "en"), ("es", "en")]
    type_of_comp_enc = {0: "Layer Norm", 1: "Attn/FC"}
    layer_to_comp = {"encoder": [layers_enc_norm, layers_enc_fc],
                     "decoder": [layers_dec_norm, layers_dec_fc]}

    for model1_name, model1 in models_to_comp:
        for model2_name, model2 in models_to_comp:
            if model1_name == model2_name:
                continue
            for pair in lang_pairs:
                src_lang, tgt_lang = pair
                dataloader_val = get_comparison_set(src_lang, tgt_lang, "mbart", split="validation[0:128]")
                for block_name, layers in layer_to_comp.items():
                    l1, l2 = layers
                    for i, l_t in enumerate(layers):
                        with ClearCache():
                            with torch.no_grad():
                                print(
                                    f"Comparing {model1_name} vs {model2_name} {type_of_comp_enc[i]} {block_name} {src_lang}->{tgt_lang}")
                                cka = paper_cka.CKA(model1, model2, model1_name, model2_name,
                                                    model1_layers=l_t,
                                                    model2_layers=l_t, device="cuda:0")
                                cka.compare(dataloader_val, debiased=True)
                                cka.plot_results(show_ticks_labels=True, short_tick_labels_splits=2,
                                                 title=f"mBART {model1_name} vs {model2_name} {type_of_comp_enc[i]} {block_name} {src_lang}->{tgt_lang}",
                                                 save_path=f"cka_imgs/mbart/{block_name}",
                                                 show_annotations=len(l_t) <= 12,
                                                 show_img=True)
