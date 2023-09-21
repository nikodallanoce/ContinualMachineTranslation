import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from typing import Dict, List, Any
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import MBartConfig, MBartForConditionalGeneration
from utilities.models import get_all_mbart_models, get_all_mt6_models
import sys
import numpy as np

sys.path.insert(0, '/home/n.dallanoce/PyCharm/pretraining')
from CosineSim import CosineSim
from utilities.models import ClearCache


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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--lang', type=str, required=True)
    args = parser.parse_args()

    if args.lang not in ["en", "fr", "de", "es"]:
        raise Exception("Lang not allowed")
    lang = args.lang

    if args.model == "mbart":
        models = get_all_mbart_models()
        src_tok = lang + "_XX" if lang != "de" else "de_DE"
        tok = AutoTokenizer.from_pretrained("nikodallanoce/mbart-cc4-vanilla-32k-5", src_lang=src_tok,
                                            tgt_lang="en_XX")
    elif args.model == "mt6":
        models = get_all_mt6_models()
        tok = AutoTokenizer.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5")
    else:
        raise Exception("Select either mbart or mt6")

    from tqdm import tqdm

    model_lst = [(k, v) for k, v in models.items() if k in ["M1",
                                                            "M2",
                                                            "M2_replay",
                                                            "M2_de_only"]]

    similarity_matrix = np.zeros((len(model_lst), len(model_lst)), dtype=float)

    with torch.no_grad():

        fn_kwargs = {'tokenizer': tok, 'lang1': lang, 'lang2': "fr"}
        if lang != "en":
            fn_kwargs['lang2'] = "en"
        wmt14_ds = get_wmt_dataset(fn_kwargs['lang1'] + "-" + fn_kwargs['lang2'], num_of_rows=256)
        dataloader = create_dataloader(wmt14_ds, "translation", fn_kwargs, 32)
        for i in range(len(model_lst) - 1):
            mi_name, model_i = model_lst[i]
            for j in range(i + 1, len(model_lst)):
                with ClearCache():
                    mj_name, model_j = model_lst[j]
                    cs = CosineSim(model_i, model_j)
                    sim = cs.compute_enc_hidd_states(dataloader, show_tqdm=False)
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
                    print(f"Similarity between {mi_name} and {mj_name} for language {lang} is {round(sim, 4)}")

    print(similarity_matrix)
    with open(f"encod_sim/{args.model} {lang}.txt", "w+") as out_file:
        out_file.write("\\begin{tabular}{|*{" + str(len(model_lst)) + "}{c|}} \\cline{1-2} \n")
        end_line = ""
        for i in range(1, len(model_lst)):
            mi_name, model_i = model_lst[i]
            end_line += f"& {mi_name}"
            line = f"${mi_name}$ "
            for j in range(0, i):
                mj_name, model_j = model_lst[j]
                line += f"& {similarity_matrix[i, j]} "
            if i == len(model_lst) - 1:
                line += "\\\\ \\hline"
            else:
                line += "\\\\ \\cline " + "{1-" + f"{str(i + 2)}" + "}"
            out_file.write(line + "\n")
        end_line = end_line + " \\\\ \\hline"
        out_file.write(end_line + "\n")
        out_file.write("\\end{tabular}")
