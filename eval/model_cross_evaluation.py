import datasets
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import MBartTokenizerFast, MBartForConditionalGeneration, MT5ForConditionalGeneration, PreTrainedModel
from typing import Set, Tuple, List, Dict, Union, Type

from bleu_utility import compute_bleu_auto_model

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


if __name__ == '__main__':
    # mbart_tok = MBartTokenizerFast.from_pretrained("nikodallanoce/mbart-cc4-vanilla-32k-5")
    testing_langs: Set[str] = {"en-fr", "en-de", "en-es"}
    both_directions: bool = True

    pre_trained_paths: Set[str] = {
        "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_ft_en-fr(MF1)/checkpoint-100000",
        "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_pre_de_ft_en-fr(MF1-2)/checkpoint-100000",
        "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_ft_en-de(MF2)/checkpoint-90000",
        "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_pre_es_ft_en-de(MF2-3)/checkpoint-100000",
        "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_ft_en-es(MF3)/checkpoint-100000",
        "/home/n.dallanoce/PyCharm/pretraining/weights/S2_mbart_pre_es_ft_en-fr(MF3-1)/checkpoint-100000"
    }

    models_lst = build_models(pre_trained_paths, MBartForConditionalGeneration)
    test_set_lst = get_all_wmt_ds(testing_langs)

    eval_results: Dict[str, Dict[str, float]] = {}

    for model in tqdm(models_lst):
        model = model.to(dev)
        ris: Dict[str, float] = {}
        model_path: str = model.config.name_or_path
        for lang_cfg, ds in tqdm(zip(testing_langs, test_set_lst)):
            langs: List[str] = lang_cfg.split("-")
            src_lang, tgt_lang = langs[0], langs[1]
            key = src_lang + "->" + tgt_lang
            ris[key] = compute_bleu_auto_model(ds, model, src_lang=src_lang, tgt_lang=tgt_lang,
                                               batch_size=32, bleu_type="bleu")["bleu"]
            if both_directions:
                key = tgt_lang + "->" + src_lang
                ris[key] = compute_bleu_auto_model(ds, model, src_lang=tgt_lang, tgt_lang=src_lang,
                                                   batch_size=32, bleu_type="bleu")["bleu"]
        eval_results[model_path] = ris

    print(eval_results)
