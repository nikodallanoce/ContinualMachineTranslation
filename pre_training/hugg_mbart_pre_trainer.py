import time
from functools import partial

import datasets
import torch.utils.data
from datasets import load_dataset, interleave_datasets
from torch.utils.data import ConcatDataset, RandomSampler
from transformers import Seq2SeqTrainingArguments, MBartConfig, \
    MBartForConditionalGeneration, MBartTokenizerFast, EarlyStoppingCallback
import os

# set the wandb project where this run will be logged
project_name = "mbart_pre_en-fr-de-es"
os.environ["WANDB_PROJECT"] = project_name

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "false"

# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys

sys.path.insert(0, '/home/n.dallanoce/PyCharm/pretraining')
from custom_datasets.MBartPreTrainingDataset import MBartPreTrainingDataset, get_item_for_iterable
from custom_datasets.RandomReplayDataset import RandomReplayDataset
from trainers.MBartTrainer import MBartTrainer
from continual.cl_tools import get_buffer, CLSampler

if __name__ == '__main__':
    size = str(int(2 ** 24))
    tok_name = "nikodallanoce/mbart-cc4-vanilla-32k-5"
    # pre_train_ds = load_dataset("text", data_files={"train": ["D:\\datasets\\test_hugg_en\\test_data_hugg.txt"]},
    #                             cache_dir="D:\\datasets\\test_hugg_en", split=f'train[0:128]',
    #                             ignore_verifications=True)
    # translation_ds = load_dataset("g8a9/europarl_en-it",
    #                             cache_dir="/data/n.dallanoce/europarl", split=f"train",
    #                             ignore_verifications=True)

    training_args = Seq2SeqTrainingArguments(f"/home/n.dallanoce/PyCharm/pretraining/weights/{project_name}_240k-steps",
                                             overwrite_output_dir=True,
                                             label_names=['labels'],
                                             do_train=True,
                                             learning_rate=1e-4,
                                             optim="adamw_torch",
                                             # auto_find_batch_size=True,
                                             per_device_train_batch_size=128,
                                             gradient_accumulation_steps=1,
                                             # num_train_epochs=1,
                                             max_steps=int(2.4e5),
                                             logging_steps=500,
                                             save_steps=10000,
                                             evaluation_strategy="steps",
                                             eval_steps=10000,
                                             logging_first_step=False,
                                             log_level="info",
                                             save_strategy="steps",
                                             fp16=True,
                                             dataloader_drop_last=True,
                                             dataloader_pin_memory=True,
                                             dataloader_num_workers=4,
                                             metric_for_best_model="pretraining_avg",
                                             greater_is_better=False,
                                             warmup_steps=0,
                                             save_total_limit=1,
                                             report_to=["wandb"],
                                             load_best_model_at_end=True
                                             )

    cc100_en = load_dataset("cc100", lang="en",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:40000000]",
                            verification_mode='no_checks')
    cc100_fr = load_dataset("cc100", lang="fr",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:40000000]",
                            verification_mode='no_checks')
    cc100_es = load_dataset("cc100", lang="es",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:40000000]",
                            verification_mode='no_checks')
    cc100_de = load_dataset("cc100", lang="de",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:40000000]",
                            verification_mode='no_checks')

    tok_en = MBartTokenizerFast.from_pretrained(tok_name, src_lang="en_XX")
    tok_fr = MBartTokenizerFast.from_pretrained(tok_name, src_lang="fr_XX")
    tok_es = MBartTokenizerFast.from_pretrained(tok_name, src_lang="es_XX")
    tok_de = MBartTokenizerFast.from_pretrained(tok_name, src_lang="de_DE")

    en_pre_train_ds = MBartPreTrainingDataset(cc100_en, tok_en, input_max_length=128)
    fr_pre_train_ds = MBartPreTrainingDataset(cc100_fr, tok_fr, input_max_length=128)
    es_pre_train_ds = MBartPreTrainingDataset(cc100_es, tok_es, input_max_length=128)
    de_pre_train_ds = MBartPreTrainingDataset(cc100_de, tok_de, input_max_length=128)

    # train from iterable dataset
    en_mc4 = load_dataset("mc4", "en", split="train", streaming=True)
    fr_mc4 = load_dataset("mc4", "fr", split="train", streaming=True)
    en_mc4 = en_mc4.map(partial(get_item_for_iterable, tokenizer=tok_en), input_columns=['text'], batched=True,
                        batch_size=2048)
    fr_mc4 = fr_mc4.map(partial(get_item_for_iterable, tokenizer=tok_fr), input_columns=['text'], batched=True,
                        batch_size=2048)
    # pre_ds = interleave_datasets([en_mc4, fr_mc4])
    # pre_train_ds = pre_ds

    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                               encoder_attention_heads=8, decoder_attention_heads=8,
                               d_model=512, max_length=128, vocab_size=tok_en.vocab_size, dropout=0.1)

    model: MBartForConditionalGeneration = MBartForConditionalGeneration(mbart_config)
    # model: MBartForConditionalGeneration = MBartForConditionalGeneration.from_pretrained(
    #     "/home/n.dallanoce/PyCharm/pretraining/weights/M2_mbart_pre_en-fr_de/checkpoint-120000")

    cc100_en_val = load_dataset("cc100", lang="en",
                                cache_dir="/data/n.dallanoce/cc100/huggingface",
                                split=f"train[60000000:60020000]",
                                verification_mode='no_checks')
    val_en_pre_train = MBartPreTrainingDataset(cc100_en_val, tok_en, input_max_length=128)

    cc100_fr_val = load_dataset("cc100", lang="fr",
                                cache_dir="/data/n.dallanoce/cc100/huggingface",
                                split=f"train[60000000:60020000]",
                                verification_mode='no_checks')
    val_fr_pre_train = MBartPreTrainingDataset(cc100_fr_val, tok_fr, input_max_length=128)

    cc100_de_val = load_dataset("cc100", lang="de",
                                cache_dir="/data/n.dallanoce/cc100/huggingface",
                                split=f"train[60000000:60020000]",
                                verification_mode='no_checks')
    val_de_pre_train = MBartPreTrainingDataset(cc100_de_val, tok_de, input_max_length=128)

    cc100_es_val = load_dataset("cc100", lang="es",
                                cache_dir="/data/n.dallanoce/cc100/huggingface",
                                split=f"train[60000000:60020000]",
                                verification_mode='no_checks')
    val_es_pre_train = MBartPreTrainingDataset(cc100_es_val, tok_es, input_max_length=128)

    pre_train_ds = torch.utils.data.ConcatDataset([en_pre_train_ds, fr_pre_train_ds, de_pre_train_ds, es_pre_train_ds])
    # curr_exp_ds = fr_pre_train_ds
    # buffer = get_buffer(prev_exp_ds=[en_pre_train_ds, fr_pre_train_ds, de_pre_train_ds],
    #                    total_cumulative_size=len(fr_pre_train_ds) * 4)
    # pre_train_ds = ConcatDataset([curr_exp_ds, buffer])

    batch_sampler = None  # CLSampler((RandomSampler(curr_exp_ds), RandomSampler(buffer)), curr_exp_frac=0.8)
    trainer = MBartTrainer(model, training_args,
                           train_dataset=pre_train_ds,
                           eval_dataset={"pretraining": ConcatDataset(
                               [val_en_pre_train, val_fr_pre_train, val_de_pre_train, val_es_pre_train])},
                           batch_sampler=batch_sampler,
                           callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
                           )
    trainer.train(resume_from_checkpoint=False)
