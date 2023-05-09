from typing import List, Dict, Union, Any

import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline, MBartConfig, MBartForConditionalGeneration, MBartTokenizer, \
    MarianTokenizer, MarianMTModel, PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, MT5ForConditionalGeneration, \
    AutoModel
from datasets import Dataset
import os

PREFIX_TASK = {'en': "English", 'fr': "French", 'de': "German", 'es': "Spanish"}


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def tokenize(examples: List[Dict[str, str]], **kwargs):
    tokenizer: MBartTokenizer = kwargs['tokenizer']
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
                        max_length=128, padding='max_length', return_attention_mask=False,
                        return_tensors='pt')

    return {'input_ids': outputs['input_ids'], 'original_text': batch_tgt}


def compute_bleu_mbart(trans_pair_ds: Dataset,
                       model: MBartForConditionalGeneration,
                       src_lang: str,
                       tgt_lang: str,
                       input_column: str = "translation",
                       max_length: int = 128,
                       num_beams: int = 5,
                       device: str = "cuda:0"):
    src_tok = src_lang + "_XX"
    tgt_tok = tgt_lang + "_XX"
    if src_lang == "de":
        src_tok = src_lang + "_DE"
    if tgt_lang == "de":
        tgt_tok = tgt_lang + "_DE"
    tok_name = "nikodallanoce/mbart-cc4-vanilla-32k-5"
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tok_name, src_lang=src_tok, tgt_lang=tgt_tok)
    fn_kwargs = {'tokenizer': tokenizer, 'lang1': src_lang, 'lang2': tgt_lang}
    test_loader: DataLoader = create_dataloader(trans_pair_ds, input_column, fn_kwargs)
    decoder_start_token_id: int = tokenizer.convert_tokens_to_ids(tgt_tok)
    results = compute_hugg_bleu(decoder_start_token_id, device, max_length, model, num_beams, test_loader, tokenizer)
    return results


def compute_bleu_mt6(trans_pair_ds: Dataset,
                     model: Union[MT5ForConditionalGeneration],
                     src_lang: str,
                     tgt_lang: str,
                     input_column: str = "translation",
                     max_length: int = 128,
                     num_beams: int = 5,
                     device: str = "cuda:0"):
    tok_name = "nikodallanoce/mt5-cc4-vanilla-32k-5"
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tok_name)
    fn_kwargs = {'tokenizer': tokenizer, 'lang1': src_lang, 'lang2': tgt_lang,
                 'task': f"translate {PREFIX_TASK[src_lang]} to {PREFIX_TASK[tgt_lang]}: "}
    test_loader: DataLoader = create_dataloader(trans_pair_ds, input_column, fn_kwargs)
    decoder_start_token_id: int = tokenizer.pad_token_id
    results = compute_hugg_bleu(decoder_start_token_id, device, max_length, model, num_beams, test_loader, tokenizer)
    return results


def compute_hugg_bleu(decoder_start_token_id, device, max_length, model, num_beams, test_loader, tokenizer):
    bleu = evaluate.load("bleu")
    for i, batch in enumerate(tqdm(test_loader)):
        translation = model.generate(batch['input_ids'].to(device), num_beams=num_beams, max_length=max_length,
                                     decoder_start_token_id=decoder_start_token_id)
        bleu.add_batch(predictions=tokenizer.batch_decode(translation, skip_special_tokens=True),
                       references=[[elem] for elem in batch['original_text']])
    results = bleu.compute()
    return results


def create_dataloader(trans_pair_ds: Dataset, input_column: str, fn_kwargs: Dict[str, Any]):
    trans_pair_ds = trans_pair_ds.map(tokenize, batched=True, input_columns=[input_column],
                                      fn_kwargs=fn_kwargs)
    trans_pair_ds = trans_pair_ds.with_format('torch', columns=["input_ids", "original_text"])
    test_loader = DataLoader(trans_pair_ds, num_workers=2, batch_size=8, drop_last=False, pin_memory=True)
    return test_loader


def compute_bleu_auto_model(trans_pair_ds: Dataset,
                            model: Union[MT5ForConditionalGeneration, MBartForConditionalGeneration],
                            src_lang: str,
                            tgt_lang: str,
                            input_column: str = "translation",
                            max_length: int = 128,
                            num_beams: int = 5,
                            device: str = "cuda:0"):
    if isinstance(model, MBartForConditionalGeneration):
        return compute_bleu_mbart(trans_pair_ds, model, src_lang, tgt_lang, input_column, max_length, num_beams, device)
    else:
        return compute_bleu_mt6(trans_pair_ds, model, src_lang, tgt_lang, input_column, max_length, num_beams, device)


#
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
    model = MT5ForConditionalGeneration.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/stream_mt5_ft_en-fr/checkpoint-30000").to(dev)

    translation_ds = load_dataset("wmt14", "fr-en",
                                  cache_dir="/data/n.dallanoce/wmt14",
                                  split=f"validation",
                                  verification_mode='no_checks')

    # translation_ds = load_dataset("nikodallanoce/wmt10", "de-en",
    #                               cache_dir="/data/n.dallanoce/wmt10",
    #                               split=f"validation",
    #                               use_auth_token=True,
    #                               verification_mode='no_checks')

    # print(len(translation_ds))
    bleu = compute_bleu_auto_model(translation_ds, model, src_lang="en", tgt_lang="fr", device=dev, num_beams=5)
    print(bleu)
#
# # pipe = pipeline(model=model, task="translation", tokenizer=tok_en, device="cuda:0")
# # text_translated = pipe("How are you?", lang1="en_XX", lang2="fr_XX")
