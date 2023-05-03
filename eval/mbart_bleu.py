from typing import List, Dict

import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline, MBartConfig, MBartForConditionalGeneration, MBartTokenizer, \
    MarianTokenizer, MarianMTModel, PreTrainedTokenizer, PreTrainedModel, AutoTokenizer
from datasets import Dataset
import os


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def tokenize(examples: List[Dict[str, str]], **kwargs):
    tokenizer: MBartTokenizer = kwargs['tokenizer']
    src_lang: str = kwargs['lang1']
    tgt_lang: str = kwargs['lang2']
    batch_src: List[str] = [e[src_lang] for e in examples]
    batch_tgt: List[str] = [e[tgt_lang] for e in examples]
    # tokenize the batch of sentences
    outputs = tokenizer(batch_src, return_special_tokens_mask=False,
                        add_special_tokens=True, truncation=True,
                        max_length=128, padding='max_length', return_attention_mask=False,
                        return_tensors='pt')

    return {'input_ids': outputs['input_ids'], 'original_text': batch_tgt}


def compute_bleu(trans_pair_ds: Dataset,
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
    tokenizer = AutoTokenizer.from_pretrained(tok_name, src_lang=src_tok, tgt_lang=tgt_tok)
    trans_pair_ds = trans_pair_ds.map(tokenize, batched=True, input_columns=[input_column],
                                      fn_kwargs={'tokenizer': tokenizer, 'lang1': src_lang, 'lang2': tgt_lang})
    trans_pair_ds = trans_pair_ds.with_format('torch', columns=["input_ids", "original_text"])
    test_loader = DataLoader(trans_pair_ds, num_workers=2, batch_size=32, drop_last=False, pin_memory=True)
    decoder_start_token_id: int = tokenizer.convert_tokens_to_ids(tgt_tok)
    bleu = evaluate.load("bleu")
    for i, batch in enumerate(tqdm(test_loader)):
        translation = model.generate(batch['input_ids'].to(device), num_beams=num_beams, max_length=max_length,
                                     decoder_start_token_id=decoder_start_token_id)
        bleu.add_batch(predictions=tokenizer.batch_decode(translation, skip_special_tokens=True),
                       references=[[elem] for elem in batch['original_text']])
        # transl += tokenizer.batch_decode(translation, skip_special_tokens=True)
        # original_txt += [[elem] for elem in batch['original_text']]

    results = bleu.compute()  # bleu.compute(predictions=transl, references=original_txt)
    return results


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
    model: MBartForConditionalGeneration = MBartForConditionalGeneration.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_ft_e1_en-fr-de/checkpoint-520000").to(dev)

    translation_ds = load_dataset("wmt14", "de-en",
                                  cache_dir="/data/n.dallanoce/wmt14",
                                  split=f"validation",
                                  verification_mode='no_checks')

    print(len(translation_ds))
    bleu = compute_bleu(translation_ds, model, src_lang="en", tgt_lang="de", device=dev, num_beams=5)
    print(bleu)
#
# # pipe = pipeline(model=model, task="translation", tokenizer=tok_en, device="cuda:0")
# # text_translated = pipe("How are you?", lang1="en_XX", lang2="fr_XX")
