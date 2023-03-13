import torch.nn
from datasets import load_dataset
import transformers
from transformers import Seq2SeqTrainingArguments, MT5Tokenizer, MT5ForConditionalGeneration, MT5Config

import sys

sys.path.insert(0, '/home/n.dallanoce/PyCharm/pretraining')
from custom_datasets.MT6PreTrainingDataset import MT6PreTrainingDataset
from trainers.MT6Trainer import MT6Trainer
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ[
    "NEPTUNE_API_TOKEN"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMmU0YTFmMy1lMDNlLTRiY2EtOTliMy02M2E3OTg4NWUzNjkifQ=="
os.environ["NEPTUNE_PROJECT"] = "nikodallanoce/mt6"


def run_local():
    pre_train_ds = load_dataset("text", data_files={"train": ["D:\\datasets\\test_hugg_en\\test_data_hugg.txt"]},
                                cache_dir="D:\\datasets\\test_hugg_en", split=f'train[0:1024]',
                                ignore_verifications=True)
    training_args = Seq2SeqTrainingArguments("D:\\trainer\\mt6",
                                             overwrite_output_dir=True,
                                             label_names=['labels'],
                                             do_train=True,
                                             # auto_find_batch_size=True,
                                             per_device_train_batch_size=2,
                                             gradient_accumulation_steps=1,
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
    return training_args, pre_train_ds


def run_server():
    training_args = Seq2SeqTrainingArguments("/home/n.dallanoce/PyCharm/pretraining/weights/mt6",
                                             overwrite_output_dir=True,
                                             label_names=['labels'],
                                             do_train=True,
                                             per_device_train_batch_size=8,
                                             gradient_accumulation_steps=1,
                                             num_train_epochs=150,
                                             optim="adamw_torch",
                                             learning_rate=1e-3,
                                             lr_scheduler_type="linear",
                                             # max_steps=int(5e5),
                                             logging_steps=100,
                                             save_steps=1600,
                                             save_strategy="steps",
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
    pre_train_ds = load_dataset("cc100", lang="en",
                                cache_dir="/data/n.dallanoce/cc100/hugg_en",
                                split=f"train[{0}:{1024}]",
                                ignore_verifications=True)
    return training_args, pre_train_ds


if __name__ == '__main__':
    size = str(int(2 ** 24))

    # special_tokens = ["en_XX", "de_DE", "es_XX", "fr_XX"]
    # tok_en = MT5Tokenizer.from_pretrained("google/mt5-base", additional_special_tokens=special_tokens)
    tok_en = MT5Tokenizer.from_pretrained("google/mt5-base")
    ris = tok_en("<extra_id_0> <extra_id_1> <extra_id_3> en_XX")

    model = MT5ForConditionalGeneration(
        MT5Config(num_layers=6, d_model=256, d_ff=1024, num_heads=4, d_kv=256 // 4, vocab_size=len(tok_en)))

    training_args, pre_train_ds = run_local()
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
