import datasets
import torch.utils.data
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, MBartConfig, \
    MBartForConditionalGeneration, MBartTokenizerFast
import os
# set the wandb project where this run will be logged
project_name = "mbart_pre_en-fr_de"
os.environ["WANDB_PROJECT"]= project_name

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys

sys.path.insert(0, '/home/n.dallanoce/PyCharm/pretraining')
from custom_datasets.MBartPreTrainingDataset import MBartPreTrainingDataset
from trainers.MBartTrainer import MBartTrainer

if __name__ == '__main__':
    size = str(int(2 ** 24))
    cc100_en = load_dataset("cc100", lang="en",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:30000000]",
                            verification_mode='no_checks')
    tok_name = "nikodallanoce/mbart-cc4-vanilla-32k-5"  # "facebook/mbart-large-cc25"
    # pre_train_ds = load_dataset("text", data_files={"train": ["D:\\datasets\\test_hugg_en\\test_data_hugg.txt"]},
    #                             cache_dir="D:\\datasets\\test_hugg_en", split=f'train[0:128]',
    #                             ignore_verifications=True)
    # translation_ds = load_dataset("g8a9/europarl_en-it",
    #                             cache_dir="/data/n.dallanoce/europarl", split=f"train",
    #                             ignore_verifications=True)
    tok_en = MBartTokenizerFast.from_pretrained(tok_name, src_lang="en_XX")
    tok_fr = MBartTokenizerFast.from_pretrained(tok_name, src_lang="fr_XX")
    tok_es = MBartTokenizerFast.from_pretrained(tok_name, src_lang="es_XX")
    tok_de = MBartTokenizerFast.from_pretrained(tok_name, src_lang="de_DE")
    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                               encoder_attention_heads=8, decoder_attention_heads=8,
                               d_model=512, max_length=128, vocab_size=tok_en.vocab_size, dropout=0.1)
    #model: MBartForConditionalGeneration = MBartForConditionalGeneration(mbart_config)
    model: MBartForConditionalGeneration = MBartForConditionalGeneration.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_pre_en-fr/checkpoint-100000")

    training_args = Seq2SeqTrainingArguments(f"/home/n.dallanoce/PyCharm/pretraining/weights/{project_name}/",
                                             overwrite_output_dir=True,
                                             label_names=['labels'],
                                             do_train=True,
                                             learning_rate=1e-4,#8e-5
                                             optim="adamw_torch",
                                             # auto_find_batch_size=True,
                                             per_device_train_batch_size=128,
                                             gradient_accumulation_steps=1,
                                             #num_train_epochs=1,
                                             max_steps=int(1e5),
                                             logging_steps=500,
                                             save_steps=5000,
                                             log_level="info",
                                             save_strategy="steps",
                                             fp16=True,
                                             dataloader_drop_last=True,
                                             dataloader_pin_memory=True,
                                             dataloader_num_workers=6,
                                             # prediction_loss_only=True,
                                             save_total_limit=1,
                                             report_to=["wandb"]
                                             )
    # training_args = Seq2SeqTrainingArguments("D:\\trainer\\mbart",
    #                                          overwrite_output_dir=True,
    #                                          label_names=['labels'],
    #                                          do_train=True,
    #                                          per_device_train_batch_size=4,
    #                                          num_train_epochs=10,
    #                                          max_steps=-1,
    #                                          log_level="debug",
    #                                          save_strategy="epoch",
    #                                          fp16=True,
    #                                          dataloader_drop_last=True,
    #                                          dataloader_pin_memory=True,
    #                                          dataloader_num_workers=1,
    #                                          # prediction_loss_only=True,
    #                                          save_total_limit=1,
    #                                          metric_for_best_model="loss",
    #                                          greater_is_better=False,
    #                                          report_to=["tensorboard"]
    #                                          )
    cc100_fr = load_dataset("cc100", lang="fr",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:30000000]",
                            verification_mode='no_checks')
    cc100_es = load_dataset("cc100", lang="es",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:30000000]",
                            verification_mode='no_checks')
    cc100_de = load_dataset("cc100", lang="de",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:30000000]",
                            verification_mode='no_checks')
    # optimizer = Adam(model.parameters(), eps=1e-6, betas=(0.9, 0.98))
    # optimizer = Adam(model.parameters())
    # lr_scheduler = transformers.get_constant_schedule(optimizer)
    # lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_training_steps=43740, num_warmup_steps=0)
    # lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_training_steps=500000, num_warmup_steps=0)
    # cc100_fr.remove_columns(["id"])
    # cc100_en.remove_columns(["id"])
    # cc100_en = cc100_en.add_column("lang1", ["en"] * len(cc100_en))
    # cc100_fr = cc100_fr.add_column("lang1", ["fr"] * len(cc100_fr))
    en_pre_train_ds = MBartPreTrainingDataset(cc100_en, tok_en, input_max_length=128)
    fr_pre_train_ds = MBartPreTrainingDataset(cc100_fr, tok_fr, input_max_length=128)
    es_pre_train_ds = MBartPreTrainingDataset(cc100_es, tok_es, input_max_length=128)
    de_pre_train_ds = MBartPreTrainingDataset(cc100_de, tok_de, input_max_length=128)
    #pre_train_ds = torch.utils.data.ConcatDataset([en_pre_train_ds, fr_pre_train_ds])
    pre_train_ds = de_pre_train_ds
    trainer = MBartTrainer(model, training_args,
                           train_dataset=pre_train_ds,
                           # optimizers=(optimizer, lr_scheduler)
                           )
    trainer.train(resume_from_checkpoint=False)