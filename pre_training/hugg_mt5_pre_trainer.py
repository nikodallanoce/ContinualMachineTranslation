import copy
import time
from functools import partial

import datasets
import torch

# torch.backends.cudnn.benchmark = True
from datasets import load_dataset
import transformers
from torch.utils.data import ConcatDataset, RandomSampler
from transformers import Seq2SeqTrainingArguments, MT5ForConditionalGeneration, MT5Config, MT5TokenizerFast, \
    T5ForConditionalGeneration, T5Config, AutoTokenizer, AutoModelForSeq2SeqLM, MBartConfig, \
    MBartForConditionalGeneration
import sys
import os

sys.path.insert(0, str(os.getcwd()))
from custom_datasets.MT6TranslationDataset import MT6TranslationDataset
from MT6TokenizerFast import MT6TokenizerFast
from iterable_datasets.IterMT6PreTrainingDataset import IterMT6PreTrainingDataset
from MT6 import MT6
from noise_functions.MT5NoiseFunction import MT5NoiseFunction
from noise_functions.MT6NoiseFunction import MT6NoiseFunction
from custom_datasets.MT6PreTrainingDataset import MT6PreTrainingDataset, get_item_for_iterable
from trainers.MT6Trainer import MT6Trainer, TrainingStrategy
from continual.cl_tools import get_buffer, CLSampler

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
project_name = "mt5_pre_en-fr(M1)"
os.environ["WANDB_PROJECT"] = project_name

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "false"

# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"


def run_server():
    training_args = Seq2SeqTrainingArguments(
        f"/data/n.dallanoce/weights/{project_name}_t5_oscar",
        overwrite_output_dir=True,
        label_names=['labels'],
        do_train=True,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=2,
        # num_train_epochs=1,
        optim="adamw_torch",
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        max_steps=int(1.8e5),
        logging_steps=500,
        save_steps=5000,
        evaluation_strategy="steps",
        eval_steps=5000,
        logging_first_step=False,
        save_strategy="steps",
        log_level="info",
        fp16=True,  # TO CHANGE
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        # load_best_model_at_end=True,
        # prediction_loss_only=True,
        save_total_limit=1,
        metric_for_best_model="pretraining_avg",
        greater_is_better=False,
        report_to=["wandb"],
        save_safetensors=False,
        torch_compile=torch.__version__.startswith("2")
    )
    return training_args


if __name__ == '__main__':
    max_inp_len = 128

    noise_fn = MT5NoiseFunction(0.35, 3)

    # tok = AutoTokenizer.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5")
    tok = AutoTokenizer.from_pretrained("google/t5-v1_1-small")
    training_args = run_server()

    # cc100_fr = load_dataset("cc100", lang="fr",
    #                         cache_dir="/data/n.dallanoce/cc100/huggingface",
    #                         split=f"train[:40000000]",
    #                         verification_mode='no_checks')
    # cc100_en = load_dataset("cc100", lang="en",
    #                         cache_dir="/data/n.dallanoce/cc100/huggingface",
    #                         split=f"train[:40000000]",
    #                         verification_mode='no_checks')

    cc100_fr = load_dataset("oscar", "unshuffled_deduplicated_fr", cache_dir="/data/n.dallanoce/oscar",
                            verification_mode="no_checks", split="train[:55000000]")
    cc100_en = load_dataset("oscar", "unshuffled_deduplicated_en", cache_dir="/data/n.dallanoce/oscar",
                            verification_mode="no_checks", split="train[:55000000]")
    cc100_de = load_dataset("cc100", lang="de",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:40000000]",
                            verification_mode='no_checks')
    cc100_es = load_dataset("cc100", lang="es",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:40000000]",
                            verification_mode='no_checks')

    en_pre_train_ds = MT6PreTrainingDataset(cc100_en, tok, input_max_length=max_inp_len,
                                            noise_fn=noise_fn)
    fr_pre_train_ds = MT6PreTrainingDataset(cc100_fr, tok, input_max_length=max_inp_len,
                                            noise_fn=noise_fn)
    de_pre_train_ds = MT6PreTrainingDataset(cc100_de, tok, input_max_length=max_inp_len,
                                            noise_fn=noise_fn)
    es_pre_train_ds = MT6PreTrainingDataset(cc100_es, tok, input_max_length=max_inp_len,
                                            noise_fn=noise_fn)

    pre_en_val = copy.copy(en_pre_train_ds)
    pre_en_val.dataset = load_dataset("oscar", "unshuffled_deduplicated_en", cache_dir="/data/n.dallanoce/oscar",
                                      verification_mode="no_checks", split="train[55000000:55020000]")
    # load_dataset("cc100", lang="en",
    #                               cache_dir="/data/n.dallanoce/cc100/huggingface",
    #                               split=f"train[40000000:40020000]",
    #                               verification_mode='no_checks')

    pre_fr_val = copy.copy(fr_pre_train_ds)
    pre_fr_val.dataset = load_dataset("oscar", "unshuffled_deduplicated_fr", cache_dir="/data/n.dallanoce/oscar",
                                      verification_mode="no_checks", split="train[55000000:55020000]")
    # load_dataset("cc100", lang="fr",
    #                                   cache_dir="/data/n.dallanoce/cc100/huggingface",
    #                                   split=f"train[40000000:40020000]",
    #                                   verification_mode='no_checks')
    pre_de_val = copy.copy(de_pre_train_ds)
    pre_de_val.dataset = load_dataset("cc100", lang="de",
                                      cache_dir="/data/n.dallanoce/cc100/huggingface",
                                      split=f"train[40000000:40020000]",
                                      verification_mode='no_checks')
    pre_es_val = copy.copy(es_pre_train_ds)
    pre_es_val.dataset = load_dataset("cc100", lang="es",
                                      cache_dir="/data/n.dallanoce/cc100/huggingface",
                                      split=f"train[40000000:40020000]",
                                      verification_mode='no_checks')

    # model = MT6.from_pretrained("/home/n.dallanoce/PyCharm/pretraining/weights/mt6_pre_en-fr(M1)_twe/checkpoint-100000")

    # model = T5ForConditionalGeneration(
    #     T5Config(num_layers=6, d_model=512, num_heads=8, d_ff=2048, vocab_size=len(tok), max_length=max_inp_len,
    #               tie_word_embeddings=True))
    model_conf = AutoModelForSeq2SeqLM.from_pretrained("google/t5-v1_1-small").config
    model_conf.vocab_size = len(tok)
    model_conf.max_length = max_inp_len
    model = T5ForConditionalGeneration(model_conf)
    # model = MBartForConditionalGeneration(MBartConfig(encoder_layers=6, decoder_layers=6,
    #                                                   encoder_ffn_dim=2048, decoder_ffn_dim=2048,
    #                                                   encoder_attention_heads=8, decoder_attention_heads=8,
    #                                                   d_model=512, max_length=128, vocab_size=tok.vocab_size,
    #                                                   dropout=0.1, decoder_start_token_id=tok.pad_token_id,
    #                                                   pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    #                                                   forced_eos_token_id=tok.eos_token_id))

    train_ds = ConcatDataset([en_pre_train_ds, fr_pre_train_ds])
    # curr_exp_ds = ConcatDataset([es_pre_train_ds, en_es_tsc_ds])
    # buffer = get_buffer(prev_exp_ds=[en_pre_train_ds, fr_pre_train_ds, en_fr_tsc_ds, de_pre_train_ds, en_de_tsc_ds],
    #                     total_cumulative_size=len(fr_pre_train_ds) * 4 + 3 * len(en_fr_tsc_ds))
    pre_train_ds = train_ds  # ConcatDataset([curr_exp_ds, buffer])

    batch_sampler = None  # CLSampler((RandomSampler(curr_exp_ds), RandomSampler(buffer)), curr_exp_frac=0.8)
    # time.sleep(2.35 * 60 * 60)
    trainer = MT6Trainer(TrainingStrategy.PRE_TRAINING, model, training_args,
                         train_dataset=pre_train_ds,
                         eval_dataset={"pretraining": ConcatDataset(
                             [pre_en_val, pre_fr_val])},
                         batch_sampler=batch_sampler,
                         tokenizer_name=tok.name_or_path
                         )
    trainer.train(resume_from_checkpoint=False)
