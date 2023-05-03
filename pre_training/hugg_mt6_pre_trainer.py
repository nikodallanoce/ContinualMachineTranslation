from functools import partial

import torch.nn
from datasets import load_dataset
import transformers
from transformers import Seq2SeqTrainingArguments, MT5ForConditionalGeneration, MT5Config, MT5TokenizerFast
from MT6TokenizerFast import MT6TokenizerFast
import sys

sys.path.insert(0, '/home/n.dallanoce/PyCharm/pretraining')
from iterable_datasets.IterMT6PreTrainingDataset import IterMT6PreTrainingDataset
from MT6 import MT6
from noise_functions.MT5NoiseFunction import MT5NoiseFunction
from noise_functions.MT6NoiseFunction import MT6NoiseFunction
from custom_datasets.MT6PreTrainingDataset import MT6PreTrainingDataset, get_item_for_iterable
from trainers.MT6Trainer import MT6Trainer
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
project_name = "stream_mt6_tests"
os.environ["WANDB_PROJECT"] = project_name

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "true"

# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"


def run_local():
    pre_train_ds = load_dataset("text", data_files={"train": ["D:\\datasets\\test_hugg_en\\test_data_hugg.txt"]},
                                cache_dir="D:\\datasets\\test_hugg_en", split=f'train[0:1024]',
                                verification_mode='no_checks')
    training_args = Seq2SeqTrainingArguments("D:\\trainer\\mt6",
                                             overwrite_output_dir=True,
                                             label_names=['labels'],
                                             do_train=True,
                                             optim="adamw_torch",
                                             learning_rate=1e-3,
                                             # auto_find_batch_size=True,
                                             per_device_train_batch_size=16,
                                             gradient_accumulation_steps=1,
                                             num_train_epochs=30,
                                             # max_steps=int(5e5),
                                             logging_steps=5,
                                             save_steps=50000,
                                             log_level="info",
                                             save_strategy="steps",
                                             fp16=True,
                                             dataloader_drop_last=True,
                                             dataloader_pin_memory=True,
                                             dataloader_num_workers=2,
                                             # prediction_loss_only=True,
                                             save_total_limit=1,
                                             metric_for_best_model="loss",
                                             greater_is_better=False,
                                             report_to=["tensorboard"]
                                             )
    return training_args, pre_train_ds


def run_server():
    training_args = Seq2SeqTrainingArguments(f"/home/n.dallanoce/PyCharm/pretraining/weights/{project_name}",
                                             overwrite_output_dir=True,
                                             # label_names=['labels_pnat', 'labels_transl'],
                                             do_train=True,
                                             per_device_train_batch_size=2,
                                             gradient_accumulation_steps=1,
                                             # num_train_epochs=1,
                                             optim="adamw_torch",
                                             learning_rate=1e-4,
                                             lr_scheduler_type="linear",
                                             max_steps=int(1e5),
                                             logging_steps=100,
                                             save_steps=10000,
                                             save_strategy="steps",
                                             log_level="info",
                                             fp16=True,
                                             dataloader_drop_last=True,
                                             dataloader_pin_memory=True,
                                             dataloader_num_workers=4,
                                             # load_best_model_at_end=True,
                                             # prediction_loss_only=True,
                                             save_total_limit=1,
                                             # evaluation_strategy = "steps",
                                             # eval_steps = 1600,
                                             # metric_for_best_model="loss",
                                             greater_is_better=False,
                                             report_to=["wandb"]
                                             )
    pre_train_ds = load_dataset("cc100", lang="en",
                                cache_dir="/data/n.dallanoce/cc100/huggingface",
                                split=f"train[0:30000000]",
                                # split=f"train[{4096}:{4096*2}]",
                                verification_mode='no_checks')
    return training_args, pre_train_ds


if __name__ == '__main__':
    size = str(int(2 ** 24))

    # special_tokens = ["en_XX", "de_DE", "es_XX", "fr_XX"]
    # tok_en = MT5Tokenizer.from_pretrained("google/mt5-base", additional_special_tokens=special_tokens)
    tok_en = MT6TokenizerFast.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5", src_lang="en_XX")
    tok_en_fr = MT6TokenizerFast.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5", src_lang="en_XX",
                                                 tgt_lang="fr_XX")
    ris = tok_en("<extra_id_0> <extra_id_1> <extra_id_3> en_XX")

    # model = MT5ForConditionalGeneration(
    #     MT5Config(num_layers=6, d_model=256, d_ff=1024, num_heads=4, d_kv=256 // 4, vocab_size=len(tok_en)))

    # model = MT5ForConditionalGeneration(MT5Config(num_layers=6, vocab_size=len(tok_en)))
    model = MT6(MT5Config(num_layers=6, d_model=512, num_heads=8, vocab_size=len(tok_en), max_length=128))
    training_args, pre_train_ds = run_server()

    # pre_train_ds = MT6PreTrainingDataset(pre_train_ds, tok_en, noise_fn=MT5NoiseFunction())
    en_mc4 = load_dataset("mc4", "en", split="train", streaming=True)
    en_mc4 = en_mc4.map(partial(get_item_for_iterable, tokenizer=tok_en, noise_fn=MT6NoiseFunction(return_list=True)),
                        input_columns=['text'], batched=True, batch_size=128,
                        remove_columns=['url', 'timestamp']).remove_columns(['text'])

    en_fr_ds = load_dataset("yhavinga/ccmatrix", "en-fr",
                            split="train",
                            streaming=True)
    en_fr_ds = en_fr_ds.map(
        partial(get_item_for_iterable, tokenizer=tok_en_fr,
                has_translation_pairs=True), batched=True,
        batch_size=128,
        remove_columns=['id', 'score'],
        input_columns=['translation']).remove_columns(['translation'])

    train_ds = IterMT6PreTrainingDataset([en_mc4, en_fr_ds])

    trainer = MT6Trainer(model, training_args,
                         train_dataset=train_ds,
                         # optimizers=(optimizer, lr_scheduler)
                         )
    trainer.train(resume_from_checkpoint=False)
