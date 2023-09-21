import copy
import time

import datasets
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import MBartTokenizerFast, MBartForConditionalGeneration, MT5ForConditionalGeneration, PreTrainedModel
from typing import Set, Tuple, List, Dict, Union, Type, Optional
import sys
import re

sys.path.insert(0, '/home/n.dallanoce/PyCharm/pretraining')
from bleu_utility import compute_bleu_auto_model
from utilities.models import get_all_mt6_models, get_all_mbart_models, ClearCache

dev = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_wmt_dataset(lang_pair: str) -> datasets.Dataset:
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
    ds = load_dataset(wmt14, lang_config,
                      cache_dir="/data/n.dallanoce/wmt14",
                      split=split,
                      verification_mode='no_checks')
    return ds


def get_all_wmt_ds(wmt_lang_conf: Union[List[str], Set[str]]) -> List[datasets.Dataset]:
    ds_lst = []

    for lang in wmt_lang_conf:
        ds_lst.append(get_wmt_dataset(lang))

    return ds_lst


def build_models(pre_trained_models_path: Set[str],
                 model: Type[Union[MBartForConditionalGeneration, MT5ForConditionalGeneration]]) -> List[
    Union[MBartForConditionalGeneration, MT5ForConditionalGeneration]]:
    models: List[PreTrainedModel] = []
    for path in pre_trained_models_path:
        models.append(model.from_pretrained(path))

    return models


def cross_evaluate(models_lst: Dict[str, Union[MBartForConditionalGeneration, MT5ForConditionalGeneration]],
                   testing_langs: Set[str], test_set_lst: List[datasets.Dataset], both_directions: bool,
                   remove_Mi: bool = True, save_dir: Optional[str] = None, bleu_tok: Optional[str] = None):
    if remove_Mi:
        for model_name in copy.copy(list(models_lst.keys())):
            if re.match("M[0-9]", model_name) is not None:
                del models_lst[model_name]
    eval_results = compute_bleu(both_directions, models_lst, test_set_lst, testing_langs, bleu_tok)
    if save_dir is not None:
        save_res_dict_on_file(eval_results, save_dir)
    return eval_results


def save_res_dict_on_file(res_dict: [str, Dict[str, float]], save_dir: str):
    with open(save_dir, "w+") as result_file:
        for key, val in res_dict.items():
            per_lan_pair_res = ""
            for pair_lang, v in val.items():
                per_lan_pair_res += f"{pair_lang}:{round(v, 2)},  "
            per_lan_pair_res = per_lan_pair_res.strip(",  ")
            result_file.write(f"{key} : {per_lan_pair_res} \n")


def compute_bleu(both_directions, models_lst, test_set_lst, testing_langs, bleu_tok):
    eval_results: Dict[str, Dict[str, float]] = {}
    for model_name, model in tqdm(models_lst.items()):
        with torch.no_grad():
            with ClearCache():
                model = model.to(dev)
                model.eval()
                ris: Dict[str, float] = {}
                for lang_cfg, ds in tqdm(zip(testing_langs, test_set_lst)):
                    langs: List[str] = lang_cfg.split("-")
                    src_lang, tgt_lang = langs[0], langs[1]
                    key = src_lang + "->" + tgt_lang
                    bleu = compute_bleu_auto_model(ds, model, src_lang=src_lang, tgt_lang=tgt_lang, batch_size=32,
                                                   bleu_tokenizer=bleu_tok)
                    ris[key] = bleu[next(iter(bleu))]
                    if both_directions:
                        key = tgt_lang + "->" + src_lang
                        bleu = compute_bleu_auto_model(ds, model, src_lang=tgt_lang, tgt_lang=src_lang, batch_size=32,
                                                       bleu_tokenizer=bleu_tok)
                        ris[key] = bleu[next(iter(bleu))]
                result_key = "mBart" if isinstance(model, MBartForConditionalGeneration) else "mT6"
                result_key = result_key + " " + model_name
                eval_results[result_key] = ris
    return eval_results


if __name__ == '__main__':
    # mbart_tok = MBartTokenizerFast.from_pretrained("nikodallanoce/mbart-cc4-vanilla-32k-5")
    testing_langs: Set[str] = {"en-fr", "en-de", "en-es"}
    both_directions: bool = True

    # save_res_dict_on_file({"mBart MF1-2": {"en->fr": 30.25, "en->de": 30.25, "de->en": 30.25, "fr->en": 30.25},
    #                        "mBart MF1-3": {"en->fr": 30.25, "en->de": 30.25, "de->en": 30.25, "fr->en": 30.25}},
    #                       r"cross_eval/mbart_test.txt")

    # pre_trained_paths: Set[str] = {
    #     "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_ft_en-fr(MF1)/checkpoint-100000",
    #     "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_pre_de_ft_en-fr(MF1-2)/checkpoint-100000",
    #     "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_ft_en-de(MF2)/checkpoint-90000",
    #     "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_pre_es_ft_en-de(MF2-3)/checkpoint-100000",
    #     "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_ft_en-es(MF3)/checkpoint-100000",
    #     "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_pre_es_ft_en-fr(MF3-1)/checkpoint-100000"
    # }

    # models_lst = build_models(pre_trained_paths, MBartForConditionalGeneration)
    models_lst = get_all_mt6_models()
    test_set_lst = get_all_wmt_ds(testing_langs)
    #time.sleep(40*60)
    eval_results = cross_evaluate(models_lst, testing_langs, test_set_lst, both_directions=True,
                                  save_dir=r"cross_eval/mt6_complete_intl.txt", bleu_tok="intl")

    print(eval_results)
