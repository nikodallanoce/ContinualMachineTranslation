from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, MBartTokenizer, MBartConfig, \
    MBartForConditionalGeneration, MT5Tokenizer, MT5ForConditionalGeneration, T5Config, MT5Config
from trainers.MT6Trainer import MT6Trainer
from CustomTrainer import CustomTrainer
import sys

from custom_datasets.MT6PreTrainingDataset import MT6PreTrainingDataset

sys.path.insert(0, '/home/n.dallanoce/PyCharm/pretraining')
from custom_datasets.MBartPreTrainingDataset import MBartPreTrainingDataset

import os


def run_local():
    pre_train_ds = load_dataset("text", data_files={"train": ["D:\\datasets\\test_hugg_en\\test_data_hugg.txt"]},
                                cache_dir="D:\\datasets\\test_hugg_en", split=f'train[0:128]',
                                ignore_verifications=True)
    training_args = Seq2SeqTrainingArguments("D:\\trainer\\mt6",
                                             overwrite_output_dir=True,
                                             label_names=['labels'],
                                             do_train=True,
                                             # auto_find_batch_size=True,
                                             per_device_train_batch_size=1,
                                             gradient_accumulation_steps=4,
                                             num_train_epochs=10,
                                             # max_steps=int(5e5),
                                             logging_steps=5,
                                             save_steps=300,
                                             log_level="info",
                                             save_strategy="steps",
                                             fp16=True,
                                             dataloader_drop_last=True,
                                             dataloader_pin_memory=True,
                                             dataloader_num_workers=1,
                                             # prediction_loss_only=True,
                                             save_total_limit=2,
                                             metric_for_best_model="loss",
                                             greater_is_better=False,
                                             report_to=["tensorboard"]
                                             )
    return pre_train_ds, training_args


def run_server():
    training_args = Seq2SeqTrainingArguments("/home/n.dallanoce/PyCharm/pretraining/weights/mt6",
                                             overwrite_output_dir=True,
                                             label_names=['labels'],
                                             do_train=True,
                                             per_device_train_batch_size=16,
                                             gradient_accumulation_steps=1,
                                             num_train_epochs=10,
                                             # max_steps=int(5e5),
                                             logging_steps=5,
                                             save_steps=1000,
                                             log_level="info",
                                             fp16=True,
                                             dataloader_drop_last=True,
                                             dataloader_pin_memory=True,
                                             dataloader_num_workers=4,
                                             # prediction_loss_only=True,
                                             save_total_limit=1,
                                             metric_for_best_model="loss",
                                             greater_is_better=False,
                                             report_to=["tensorboard"]
                                             )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    pre_train_ds = load_dataset("cc100", lang="en",
                                cache_dir="/data/n.dallanoce/cc100/hugg_en",
                                split=f"train[{0}:{1024}]",
                                ignore_verifications=True)
    return pre_train_ds, training_args


if __name__ == '__main__':
    size = str(int(2 ** 24))

    special_tokens = ["en_XX", "de_DE", "es_XX", "fr_XX"]
    tok_en = MT5Tokenizer.from_pretrained("google/mt5-base", additional_special_tokens=special_tokens)

    ris = tok_en("<extra_id_0> <extra_id_1> <extra_id_3> en_XX")

    special_tokens += tok_en.additional_special_tokens
    tok_en.add_special_tokens({"additional_special_tokens": special_tokens})

    model = MT5ForConditionalGeneration(MT5Config(vocab_size=len(tok_en)))
    pre_train_ds, training_args = run_local()
    # , decoder_start_token_id=tok_en.eos_token_id))

    # optimizer = Adam(model.parameters(), eps=1e-6, betas=(0.9, 0.98))
    # optimizer = Adam(model.parameters())
    # lr_scheduler = transformers.get_constant_schedule(optimizer)
    # lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_training_steps=43740, num_warmup_steps=0)
    # lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_training_steps=500000, num_warmup_steps=0)

    pre_train_ds = MT6PreTrainingDataset(pre_train_ds, tok_en)
    trainer = MT6Trainer(model, training_args,
                         train_dataset=pre_train_ds,
                         # optimizers=(optimizer, lr_scheduler)
                         )
    trainer.train(resume_from_checkpoint=False)
