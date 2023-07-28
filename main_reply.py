import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, MBartTokenizerFast

from custom_datasets.MBartPreTrainingDataset import MBartPreTrainingDataset
from custom_datasets.RandomReplayDataset import RandomReplayDataset
from custom_datasets.MBartTranslationDataset import MBartTranslationDataset

if __name__ == '__main__':
    cc100_en = load_dataset("cc100", lang="en",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:40000]",
                            verification_mode='no_checks')
    cc100_fr = load_dataset("cc100", lang="fr",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:40000]",
                            verification_mode='no_checks')
    cc100_es = load_dataset("cc100", lang="es",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:40000]",
                            verification_mode='no_checks')
    cc100_de = load_dataset("cc100", lang="de",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:40000]",
                            verification_mode='no_checks')

    tok_name = "nikodallanoce/mbart-cc4-vanilla-32k-5"
    tok_en = MBartTokenizerFast.from_pretrained(tok_name, src_lang="en_XX")
    tok_fr = MBartTokenizerFast.from_pretrained(tok_name, src_lang="fr_XX")
    tok_es = MBartTokenizerFast.from_pretrained(tok_name, src_lang="es_XX")
    tok_de = MBartTokenizerFast.from_pretrained(tok_name, src_lang="de_DE")

    en_pre_train_ds = MBartPreTrainingDataset(cc100_en, tok_en, input_max_length=128)
    fr_pre_train_ds = MBartPreTrainingDataset(cc100_fr, tok_fr, input_max_length=128)
    es_pre_train_ds = MBartPreTrainingDataset(cc100_es, tok_es, input_max_length=128)
    de_pre_train_ds = MBartPreTrainingDataset(cc100_de, tok_de, input_max_length=128)
    fr_token = tok_en.convert_tokens_to_ids("fr_XX")
    en_token = tok_en.convert_tokens_to_ids("en_XX")
    counter = 0
    total = 0
    dl = DataLoader(
        RandomReplayDataset(curr_exp_ds_lst=[de_pre_train_ds], prev_exp_ds_lst=[en_pre_train_ds, fr_pre_train_ds]),
        shuffle=True)

    progress = tqdm(dl)
    for e in progress:
        inp_ids = e['input_ids']
        total = total + 1
        if torch.sum(inp_ids[inp_ids == fr_token]) > 0 or torch.sum(inp_ids[inp_ids == en_token]) > 0:
            counter = counter + 1
            # print(tok_en.batch_decode(inp_ids))
        progress.set_postfix(ratio=counter / total)
    print(counter)
