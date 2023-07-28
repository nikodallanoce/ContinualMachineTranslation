from typing import List, Dict, Union, Any

import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBartTokenizer, \
    PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, MT5ForConditionalGeneration, \
    T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from datasets import Dataset
import os
import sys

sys.path.insert(0, '/home/n.dallanoce/PyCharm/pretraining')
from MT6TokenizerFast import MT6TokenizerFast

PREFIX_TASK = {'en': "English", 'fr': "French", 'de': "German", 'es': "Spanish"}


#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def tokenize(examples: List[Dict[str, str]], **kwargs):
    tokenizer: PreTrainedTokenizer = kwargs['tokenizer']
    src_lang: str = kwargs['lang1']
    tgt_lang: str = kwargs['lang2']
    if "task" in kwargs:
        task: str = kwargs['task']
        batch_src: List[str] = [task + e[src_lang] for e in examples]
    else:
        batch_src: List[str] = [e[src_lang] for e in examples]
    batch_tgt: List[str] = [e[tgt_lang] for e in examples]
    # tokenize the batch of sentences
    outputs = tokenizer(batch_src, return_special_tokens_mask=False,
                        add_special_tokens=True, truncation=True,
                        max_length=128, padding='max_length',
                        return_attention_mask=False, return_tensors='pt')
    # labels = tokenizer(batch_tgt, truncation=False)
    # batch_tgt = tokenizer.batch_decode(labels['input_ids'], skip_special_tokens=True)

    return {'input_ids': outputs['input_ids'], 'original_text': batch_tgt}


def compute_bleu_mbart(trans_pair_ds: Dataset,
                       model: Union[MBartForConditionalGeneration, PreTrainedModel],
                       src_lang: str,
                       tgt_lang: str,
                       input_column: str = "translation",
                       max_length: int = 128,
                       num_beams: int = 5,
                       device: str = "cuda:0",
                       batch_size: int = 32,
                       bleu_type: str = "sacrebleu",
                       bleu_tokenizer=None,
                       tokenizer_name: str = None) -> Dict[str, Any]:
    src_tok, tgt_tok = get_langs_token(src_lang, tgt_lang)
    tok_name = "nikodallanoce/mbart-cc4-vanilla-32k-5" if tokenizer_name is None else tokenizer_name
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tok_name, src_lang=src_tok, tgt_lang=tgt_tok)
    fn_kwargs = {'tokenizer': tokenizer, 'lang1': src_lang, 'lang2': tgt_lang}
    test_loader: DataLoader = create_dataloader(trans_pair_ds, input_column, fn_kwargs, batch_size)
    decoder_start_token_id: int = tokenizer.convert_tokens_to_ids(tgt_tok)
    results = compute_hugg_bleu(decoder_start_token_id, device, max_length, model, num_beams, test_loader, tokenizer,
                                bleu_type, bleu_tokenizer)
    return results


def get_langs_token(src_lang, tgt_lang):
    src_tok = src_lang + "_XX"
    tgt_tok = tgt_lang + "_XX"
    if src_lang == "de":
        src_tok = src_lang + "_DE"
    if tgt_lang == "de":
        tgt_tok = tgt_lang + "_DE"
    return src_tok, tgt_tok


def compute_bleu_mt6(trans_pair_ds: Dataset,
                     model: Union[MT5ForConditionalGeneration, PreTrainedModel],
                     src_lang: str,
                     tgt_lang: str,
                     input_column: str = "translation",
                     max_length: int = 128,
                     num_beams: int = 5,
                     device: str = "cuda:0",
                     batch_size: int = 32,
                     bleu_type: str = "sacrebleu",
                     bleu_tokenizer=None,
                     tokenizer_name: str = None,
                     use_lang_tokens: bool = False) -> Dict[str, Any]:
    tok_name = "nikodallanoce/mt5-cc4-vanilla-32k-5" if tokenizer_name is None else tokenizer_name
    # "google/t5-v1_1-small"  # "nikodallanoce/mt5-cc4-vanilla-32k-5"
    if use_lang_tokens:
        src_tok, tgt_tok = get_langs_token(src_lang, tgt_lang)
        tokenizer: PreTrainedTokenizer = MT6TokenizerFast.from_pretrained(tok_name, src_lang=src_tok, tgt_lang=tgt_tok)
        decoder_start_token_id: int = tokenizer.convert_tokens_to_ids(tgt_tok)
    else:
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tok_name)
        decoder_start_token_id: int = tokenizer.pad_token_id

    fn_kwargs = {'tokenizer': tokenizer, 'lang1': src_lang, 'lang2': tgt_lang}
    if not isinstance(tokenizer, MT6TokenizerFast):
        fn_kwargs['task'] = f"translate {PREFIX_TASK[src_lang]} to {PREFIX_TASK[tgt_lang]}: "

    test_loader: DataLoader = create_dataloader(trans_pair_ds, input_column, fn_kwargs, batch_size)
    # decoder_start_token_id = tokenizer.pad_token_id
    results = compute_hugg_bleu(decoder_start_token_id, device, max_length, model, num_beams, test_loader, tokenizer,
                                bleu_type, bleu_tokenizer)
    return results


def compute_hugg_bleu(decoder_start_token_id: int, device: str, max_length: int,
                      model: Union[MT5ForConditionalGeneration, MBartForConditionalGeneration], num_beams: int,
                      test_loader: DataLoader, tokenizer: PreTrainedTokenizer,
                      bleu_type: str, bleu_tokenizer: str) -> Dict[str, Any]:
    bleu_metric = evaluate.load(bleu_type)
    for i, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            translation = model.generate(batch['input_ids'].to(device), num_beams=num_beams, max_length=max_length,
                                         forced_bos_token_id=decoder_start_token_id)
        bleu_metric.add_batch(predictions=tokenizer.batch_decode(translation, skip_special_tokens=True),
                              references=[[elem] for elem in batch['original_text']])

    results = bleu_metric.compute() if bleu_tokenizer is None else bleu_metric.compute(tokenize=bleu_tokenizer)
    if bleu_type in results:
        results['bleu_metric'] *= 100
    return results


def create_dataloader(trans_pair_ds: Dataset, input_column: str, fn_kwargs: Dict[str, Any],
                      batch_size: int) -> DataLoader:
    trans_pair_ds = trans_pair_ds.map(tokenize, batched=True, input_columns=[input_column],
                                      fn_kwargs=fn_kwargs)
    trans_pair_ds = trans_pair_ds.with_format('torch', columns=["input_ids", "original_text"])
    test_loader = DataLoader(trans_pair_ds, num_workers=4, batch_size=batch_size, drop_last=False, pin_memory=True)
    return test_loader


def compute_bleu_auto_model(trans_pair_ds: Dataset,
                            model: Union[MT5ForConditionalGeneration, MBartForConditionalGeneration],
                            src_lang: str,
                            tgt_lang: str,
                            input_column: str = "translation",
                            max_length: int = 128,
                            num_beams: int = 5,
                            device: str = "cuda:0",
                            batch_size: int = 8,
                            bleu_type: str = "sacrebleu",
                            bleu_tokenizer: str = None,
                            tokenizer_name: str = None) -> Dict[str, Any]:
    if isinstance(model, MBartForConditionalGeneration):
        return compute_bleu_mbart(trans_pair_ds, model, src_lang, tgt_lang, input_column, max_length, num_beams, device,
                                  batch_size, bleu_type, bleu_tokenizer, tokenizer_name)
    elif isinstance(model, MT5ForConditionalGeneration):
        return compute_bleu_mt6(trans_pair_ds, model, src_lang, tgt_lang, input_column, max_length, num_beams, device,
                                batch_size, bleu_type, bleu_tokenizer, tokenizer_name)
    else:
        raise TypeError("Invalid model type.")


if __name__ == '__main__':
    #     # translation_ds = load_dataset("yhavinga/ccmatrix", "en-fr",
    #     #                               cache_dir="/data/n.dallanoce/cc_en_fr",
    #     #                               split=f"train[25000000:25003000]",
    #     #                               ignore_verifications=True)
    #
    #     # to_translate, original = translation_ds[1]['translation']['en'], translation_ds[1]['translation']['fr']
    #
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    #     # tok_en = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", lang1="en_XX", lang2="fr_XX")
    model: Union[MBartForConditionalGeneration, MT5ForConditionalGeneration] = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_ft_en-fr-Mf1_conc_trsl/checkpoint-270000").to(dev)
    # translation_ds = load_dataset("yhavinga/ccmatrix", "en-es",
    #                               cache_dir="/data/n.dallanoce/cc_en_es",
    #                               split=f"train[28000000:28003000]",
    #                               verification_mode='no_checks')
    # translation_ds = translation_ds.with_format("torch", columns=['translation'])

    # translation_ds = load_dataset("nikodallanoce/wmt10", "de-en",
    #                               cache_dir="/data/n.dallanoce/wmt10",
    #                               split=f"validation",
    #                               use_auth_token=True,
    #                               verification_mode='no_checks')

    # print(len(translation_ds))
    translation_ds = load_dataset("wmt14", "fr-en",
                                  cache_dir="/data/n.dallanoce/wmt14",
                                  split=f"test",
                                  verification_mode='no_checks')
    src_lang, tgt_lang = "en", "fr"
    
    bleu = compute_bleu_auto_model(translation_ds, model, src_lang=src_lang, tgt_lang=tgt_lang, device=dev, num_beams=5,
                                   batch_size=32, bleu_type="sacrebleu", bleu_tokenizer='intl', tokenizer_name=None)
    # bleu = compute_bleu_mt6(translation_ds, model, src_lang=src_lang, tgt_lang=tgt_lang, device=dev, num_beams=5,
    #                         batch_size=32, bleu_type="sacrebleu", bleu_tokenizer="intl",
    #                         tokenizer_name="nikodallanoce/mt6_tok_fast",
    #                         use_lang_tokens=True)
    s_k = next(iter(bleu))
    print(f"{src_lang} --> {tgt_lang}: {bleu[s_k]}")
