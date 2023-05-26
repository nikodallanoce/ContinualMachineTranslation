from functools import partial

import datasets
import torch.nn
from datasets import load_dataset
import transformers
from torch.utils.data import ConcatDataset
from transformers import Seq2SeqTrainingArguments, MT5ForConditionalGeneration, MT5Config, MT5TokenizerFast
import sys

sys.path.insert(0, '/home/n.dallanoce/PyCharm/pretraining')
from custom_datasets.MT6TranslationDataset import MT6TranslationDataset
from MT6TokenizerFast import MT6TokenizerFast
from iterable_datasets.IterMT6PreTrainingDataset import IterMT6PreTrainingDataset
from MT6 import MT6
from noise_functions.MT5NoiseFunction import MT5NoiseFunction
from noise_functions.MT6NoiseFunction import MT6NoiseFunction
from custom_datasets.MT6PreTrainingDataset import MT6PreTrainingDataset, get_item_for_iterable
from trainers.MT6Trainer import MT6Trainer
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
project_name = "mt6_pre_de_ft_en-fr(MF1-2)"
os.environ["WANDB_PROJECT"] = project_name

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "true"

# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"

if __name__ == '__main__':
    training_args = Seq2SeqTrainingArguments(f"/home/n.dallanoce/PyCharm/pretraining/weights/{project_name}",
                                             overwrite_output_dir=True,
                                             # label_names=['labels_pnat', 'labels_transl'],
                                             do_train=True,
                                             per_device_train_batch_size=128,
                                             gradient_accumulation_steps=1,
                                             # num_train_epochs=1,
                                             optim="adamw_torch",
                                             learning_rate=1e-3,
                                             lr_scheduler_type="linear",
                                             max_steps=int(1e5),
                                             logging_steps=500,
                                             save_steps=10000,
                                             save_strategy="steps",
                                             log_level="info",
                                             load_best_model_at_end=True,
                                             evaluation_strategy="steps",
                                             eval_steps=10000,
                                             fp16=True,
                                             dataloader_drop_last=True,
                                             dataloader_pin_memory=True,
                                             dataloader_num_workers=4,
                                             # prediction_loss_only=True,
                                             save_total_limit=1,
                                             metric_for_best_model="bleu_en_fr",
                                             greater_is_better=True,
                                             report_to=["wandb"],
                                             ignore_data_skip=True
                                             )

    tok_en = MT5TokenizerFast.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5")

    translation_ds = load_dataset("yhavinga/ccmatrix", "en-fr",
                                  cache_dir="/data/n.dallanoce/cc_en_fr",
                                  split=f"train[0:25000000]",
                                  verification_mode='no_checks')

    en_fr_ds = MT6TranslationDataset(translation_ds, tok_en, src_lang="en", tgt_lang="fr", input_max_length=128,
                                     noise_fn=None,
                                     skip_rows={2372581, 6968567, 10821748, 11060884, 15767927, 25424386, 29725453,
                                                45747545, 47137798, 50129051, 59177023, 59929203, 61511560, 75542580,
                                                100970169, 115986518, 127680776, 141459031, 156717917, 157018533,
                                                162558439, 164150364, 175041176, 184342700, 190148649, 190148650,
                                                192658445, 220362372, 245452855, 256201123, 271393589, 272871204,
                                                272877704, 281597372, 294584774, 296244867, 321887045})
    fr_en_ds = MT6TranslationDataset(translation_ds, tok_en, src_lang="fr", tgt_lang="en", input_max_length=128,
                                     noise_fn=None,
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

    model = MT5ForConditionalGeneration.from_pretrained(
        "/home/n.dallanoce/PyCharm/pretraining/weights/mt6_pre_en-fr_de(M2)/checkpoint-100000")

    # model = MT5ForConditionalGeneration(
    #     MT5Config(num_layers=6, d_model=512, num_heads=8, vocab_size=len(tok_en), max_length=128))

    val_ds_fr_en = load_dataset("wmt14", "fr-en",
                                cache_dir="/data/n.dallanoce/wmt14",
                                split=f"validation",
                                verification_mode='no_checks')
    val_ds_config_en_fr = val_ds_fr_en.config_name.replace("-", "_")
    trainer = MT6Trainer("finetuning", model, training_args,
                         train_dataset=ConcatDataset([en_fr_ds, fr_en_ds]),
                         eval_dataset={f"{val_ds_config_en_fr}": val_ds_fr_en}
                         )
    trainer.train(resume_from_checkpoint=False)
