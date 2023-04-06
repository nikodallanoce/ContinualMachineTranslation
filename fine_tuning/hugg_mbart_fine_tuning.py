from typing import Dict

import torch
from torch.utils.data import ConcatDataset
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, MBartTokenizer, MBartConfig, \
    MBartForConditionalGeneration, Seq2SeqTrainer, EvalPrediction, AutoTokenizer
import sys

sys.path.insert(0, '/home/n.dallanoce/PyCharm/pretraining')
from custom_datasets.MBartTranslationDataset import MBartTranslationDataset
from trainers.MBartTrainer import MBartTrainer
import os


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def compute_bleu_metric(prediction: EvalPrediction):
    eval_metrics: Dict[str, float] = dict()
    # for eval_task in self.eval_dataset:
    #     transl_pair_ds = self.eval_dataset[eval_task]
    #     src_tgt_langs = eval_task.split("_")
    #     eval_metrics[metric_key_prefix + eval_task] = \
    #         compute_bleu(transl_pair_ds, self.model, src_lang=src_tgt_langs[1],
    #                      tgt_lang=src_tgt_langs[2])["bleu"] * 100
    return eval_metrics


def run_server():
    model = MBartForConditionalGeneration.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_cc100_hilr/checkpoint-500000")
    training_args = Seq2SeqTrainingArguments("/home/n.dallanoce/PyCharm/pretraining/weights/mbart_ft_fr-en_hilr",
                                             overwrite_output_dir=True,
                                             label_names=['labels'],
                                             do_train=True,
                                             label_smoothing_factor=0,
                                             # warmup_steps=2500,
                                             optim="adamw_torch",
                                             learning_rate=4e-5,
                                             # auto_find_batch_size=True,
                                             per_device_train_batch_size=128,
                                             gradient_accumulation_steps=1,
                                             # num_train_epochs=60,
                                             max_steps=int(3e5),
                                             logging_steps=500,
                                             save_steps=5000,
                                             log_level="info",
                                             save_strategy="steps",
                                             load_best_model_at_end=True,
                                             evaluation_strategy="steps",
                                             eval_steps=5000,
                                             fp16=True,
                                             dataloader_drop_last=True,
                                             dataloader_pin_memory=True,
                                             dataloader_num_workers=4,
                                             # prediction_loss_only=True,
                                             save_total_limit=1,
                                             metric_for_best_model="bleu_en_fr",
                                             greater_is_better=True,
                                             report_to=["tensorboard"],
                                             ignore_data_skip=True
                                             )
    translation_ds = load_dataset("yhavinga/ccmatrix", "en-fr",
                                  cache_dir="/data/n.dallanoce/cc_en_fr",
                                  split=f"train[0:25000000]",
                                  verification_mode='no_checks')

    val_ds = load_dataset("wmt14", "fr-en",
                          cache_dir="/data/n.dallanoce/wmt14",
                          split=f"validation",
                          verification_mode='no_checks')

    return training_args, translation_ds, model, val_ds.with_format("torch", columns=["translation"])


def run_local():
    training_args = Seq2SeqTrainingArguments("D:\\trainer\\mbart_en_fr",
                                             overwrite_output_dir=True,
                                             label_names=['labels'],
                                             do_train=True,
                                             do_eval=True,
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
                                             save_steps=10,
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
                                             greater_is_better=True,
                                             report_to=["tensorboard"]
                                             )
    dataset = load_dataset("wmt14", "fr-en",
                           cache_dir="D:\\datasets\\wmt14",
                           split=f"train[0:2048]",
                           verification_mode='no_checks')

    val_ds = load_dataset("wmt14", "fr-en",
                          cache_dir="D:\\datasets\\wmt14",
                          split=f"validation",
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

    tok_name = "nikodallanoce/mbart-cc4-vanilla-32k-5"
    tok_en_fr = AutoTokenizer.from_pretrained(tok_name, src_lang="en_XX", tgt_lang="fr_XX")
    tok_fr_en = AutoTokenizer.from_pretrained(tok_name, src_lang="fr_XX", tgt_lang="en_XX")
    training_args, translation_ds, model, val_ds = run_server()

    en_fr_ds = MBartTranslationDataset(translation_ds, tok_en_fr, src_lang="en", trg_lang="fr", input_max_length=128,
                                       skip_rows={2372581, 6968567, 10821748, 11060884, 15767927, 25424386, 29725453,
                                                  45747545, 47137798, 50129051, 59177023, 59929203, 61511560, 75542580,
                                                  100970169, 115986518, 127680776, 141459031, 156717917, 157018533,
                                                  162558439, 164150364, 175041176, 184342700, 190148649, 190148650,
                                                  192658445, 220362372, 245452855, 256201123, 271393589, 272871204,
                                                  272877704, 281597372, 294584774, 296244867, 321887045})
    fr_en_ds = MBartTranslationDataset(translation_ds, tok_fr_en, src_lang="fr", trg_lang="en", input_max_length=128,
                                       skip_rows={35870050, 48145532, 52684654, 58751416, 58882125, 65601877, 67930837,
                                                  77241694, 92977227, 110216804, 128101180, 134271264, 141335940,
                                                  163685146, 170148774, 174846035, 175041176, 178472316, 187909576,
                                                  190148650, 190788599, 199867191, 202841440, 203367259, 216538756,
                                                  216971114, 217029239, 217343772, 221404922, 228346708, 229954946,
                                                  236475781, 238254814, 239313560, 240741477, 244758333, 244990634,
                                                  246716684, 246848239, 251633313, 252437626, 258612355, 260023316,
                                                  261848203, 266413071, 269838607, 271039088, 280243425, 290825579,
                                                  303987750, 304028810, 310067703, 310183397, 314725258, 323880921,
                                                  324665884})

    eval_tokenizers: Dict[str, MBartTokenizer] = {"en_fr": tok_en_fr, "fr_en": tok_fr_en}
    val_ds_config = val_ds.config_name.replace("-", "_")  # works with wmt14 datasets
    trainer = MBartTrainer(model, training_args,
                           train_dataset=ConcatDataset([en_fr_ds, fr_en_ds]),
                           # eval_dataset={'bleu_en_fr': val_ds, 'bleu_fr_en': val_ds},  # , 'bleu_fr_en': val_ds},
                           eval_dataset={f"{val_ds_config}": val_ds},
                           # optimizers=(optimizer, lr_scheduler)
                           )

    trainer.train(resume_from_checkpoint=False)
