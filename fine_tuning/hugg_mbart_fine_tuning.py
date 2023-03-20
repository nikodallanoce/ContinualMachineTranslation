from typing import Dict

import torch
from torch.utils.data import ConcatDataset
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, MBartTokenizer, MBartConfig, \
    MBartForConditionalGeneration, Seq2SeqTrainer
import sys

sys.path.insert(0, '/home/n.dallanoce/PyCharm/pretraining')
from custom_datasets.MBartTranslationDataset import MBartTranslationDataset
from trainers.MBartTrainer import MBartTrainer


def run_server():
    model = MBartForConditionalGeneration.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_cc100/checkpoint-100000", dropout=0.3)
    training_args = Seq2SeqTrainingArguments("/home/n.dallanoce/PyCharm/pretraining/weights/mbart_ft_fr-en_cc_2/",
                                             overwrite_output_dir=True,
                                             label_names=['labels'],
                                             do_train=True,
                                             label_smoothing_factor=0,
                                             # warmup_steps=2500,
                                             optim="adamw_torch",
                                             # learning_rate=3e-5,
                                             # auto_find_batch_size=True,
                                             per_device_train_batch_size=32,
                                             gradient_accumulation_steps=1,
                                             # num_train_epochs=60,
                                             max_steps=int(5e5),
                                             logging_steps=1000,
                                             save_steps=10000,
                                             log_level="info",
                                             save_strategy="steps",
                                             # load_best_model_at_end=True,
                                             evaluation_strategy="steps",
                                             eval_steps=10,
                                             fp16=True,
                                             dataloader_drop_last=True,
                                             dataloader_pin_memory=True,
                                             dataloader_num_workers=8,
                                             # prediction_loss_only=True,
                                             save_total_limit=1,
                                             metric_for_best_model="loss",
                                             greater_is_better=False,
                                             report_to=["tensorboard"]
                                             )
    translation_ds = load_dataset("yhavinga/ccmatrix", "en-fr",
                                  cache_dir="/data/n.dallanoce/cc_en_fr",
                                  split=f"train[0:20000000]",
                                  verification_mode='no_checks')

    val_ds = load_dataset("yhavinga/ccmatrix", "en-fr",
                          cache_dir="/data/n.dallanoce/cc_en_fr",
                          split=f"train[:-2048]",
                          verification_mode='no_checks')

    return training_args, translation_ds, model, val_ds.with_format("torch", columns=["translation"])


def run_local():
    training_args = Seq2SeqTrainingArguments("D:\\trainer\\mbart_en_fr",
                                             overwrite_output_dir=True,
                                             label_names=['labels'],
                                             do_train=True,
                                             label_smoothing_factor=0.1,
                                             warmup_steps=2500,
                                             optim="adamw_torch",
                                             learning_rate=3e-5,
                                             # auto_find_batch_size=True,
                                             per_device_train_batch_size=2,
                                             gradient_accumulation_steps=1,
                                             num_train_epochs=40,
                                             # max_steps=int(5e5),
                                             logging_steps=10,
                                             save_steps=10000,
                                             log_level="info",
                                             evaluation_strategy="steps",
                                             eval_steps=10,
                                             save_strategy="steps",
                                             fp16=True,
                                             dataloader_drop_last=True,
                                             dataloader_pin_memory=True,
                                             dataloader_num_workers=1,
                                             # prediction_loss_only=True,
                                             save_total_limit=2,
                                             metric_for_best_model="bleu_en_fr",
                                             greater_is_better=False,
                                             report_to=["tensorboard"]
                                             )
    dataset = load_dataset("wmt14", "fr-en",
                           cache_dir="D:\\datasets\\wmt14",
                           split=f"train[0:2048]",
                           verification_mode='no_checks')

    val_ds = load_dataset("wmt14", "fr-en",
                          cache_dir="D:\\datasets\\wmt14",
                          split=f"train[:-2048]",
                          verification_mode='no_checks')

    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                               encoder_attention_heads=8, decoder_attention_heads=8,
                               d_model=512, max_length=128, vocab_size=tok_en_fr.vocab_size, dropout=0.2)
    model: MBartForConditionalGeneration = MBartForConditionalGeneration(mbart_config)
    return training_args, dataset, model, val_ds.with_format("torch", columns=["translation"])


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    size = str(int(2 ** 24))
    # translation_ds = load_dataset("wmt14", "fr-en",
    #                               cache_dir="/data/n.dallanoce/wmt14",
    #                               split=f"train",
    #                               ignore_verifications=True)

    # training_args, translation_ds = run_server()
    training_args, translation_ds, model, val_ds = run_local()
    tok_en_fr = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="fr_XX")
    tok_fr_en = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="fr_XX", tgt_lang="en_XX")

    en_fr_ds = MBartTranslationDataset(translation_ds, tok_en_fr, src_lang="en", trg_lang="fr")
    fr_en_ds = MBartTranslationDataset(translation_ds, tok_fr_en, src_lang="fr", trg_lang="en")

    eval_tokenizers: Dict[str, MBartTokenizer] = {"en_fr": tok_en_fr,
                                                  "fr_en": tok_fr_en}
    trainer = MBartTrainer(model, training_args,
                           train_dataset=ConcatDataset([en_fr_ds, fr_en_ds]),
                           eval_dataset={'bleu_en_fr': val_ds, 'bleu_fr_en': val_ds},
                           eval_tokenizers=eval_tokenizers
                           # optimizers=(optimizer, lr_scheduler)
                           )

    trainer.train(resume_from_checkpoint=False)
