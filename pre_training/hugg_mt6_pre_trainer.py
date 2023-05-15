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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
project_name = "deltalm_pre_en-fr"
os.environ["WANDB_PROJECT"] = project_name

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "true"

# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"


def run_server():
    training_args = Seq2SeqTrainingArguments(f"/home/n.dallanoce/PyCharm/pretraining/weights/{project_name}",
                                             overwrite_output_dir=True,
                                             # label_names=['labels_pnat', 'labels_transl'],
                                             do_train=True,
                                             per_device_train_batch_size=128,
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
                                             dataloader_num_workers=2,
                                             # load_best_model_at_end=True,
                                             # prediction_loss_only=True,
                                             save_total_limit=1,
                                             # evaluation_strategy = "steps",
                                             # eval_steps = 1600,
                                             # metric_for_best_model="loss",
                                             greater_is_better=False,
                                             report_to=["wandb"]
                                             )
    return training_args


if __name__ == '__main__':
    size = str(int(2 ** 24))
    tok_en = MT5TokenizerFast.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5")
    model = MT6(MT5Config(num_layers=6, d_model=512, num_heads=8, vocab_size=len(tok_en), max_length=128))
    # model = MT6.from_pretrained("/home/n.dallanoce/PyCharm/pretraining/weights/mt6_pre_en-fr/checkpoint-100000")
    training_args = run_server()

    # pre_train_ds = MT6PreTrainingDataset(pre_train_ds, tok_en, noise_fn=MT5NoiseFunction())
    en_mc4 = load_dataset("mc4", languages=["en"], split="train", streaming=True)
    fr_mc4 = load_dataset("mc4", languages=["fr"], split="train", streaming=True)
    en_fr_mc4 = datasets.interleave_datasets([en_mc4, fr_mc4])
    # en_fr_mc4 = en_fr_mc4.map(partial(get_item_for_iterable, tokenizer=tok_en,
    #                                   noise_fn=MT6NoiseFunction(n_groups=2, noise_density=0.3, pnat=True)),
    #                           input_columns=['text'], batched=True, batch_size=128,
    #                           remove_columns=['url', 'timestamp']).remove_columns(['text'])
    en_fr_mc4 = en_fr_mc4.map(partial(get_item_for_iterable, tokenizer=tok_en,
                                      noise_fn=MT5NoiseFunction(noise_density=0.3)),
                              input_columns=['text'], batched=True, batch_size=256,
                              remove_columns=['url', 'timestamp']).remove_columns(['text'])

    en_fr_ds = load_dataset("yhavinga/ccmatrix", "en-fr",
                            split="train",
                            streaming=True)
    en_fr_ds = en_fr_ds.map(
        partial(get_item_for_iterable, tokenizer=tok_en,
                noise_fn=MT6NoiseFunction(n_groups=2, noise_density=0.3, pnat=True),
                has_translation_pairs=True), batched=True,
        batch_size=128,
        remove_columns=['id', 'score'],
        input_columns=['translation']).remove_columns(['translation'])

    it_train_ds = IterMT6PreTrainingDataset([en_fr_mc4])

    cc100_fr = load_dataset("cc100", lang="fr",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:10000000]",
                            verification_mode='no_checks')
    cc100_en = load_dataset("cc100", lang="en",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:10000000]",
                            verification_mode='no_checks')
    cc100_de = load_dataset("cc100", lang="de",
                            cache_dir="/data/n.dallanoce/cc100/huggingface",
                            split=f"train[:20000000]",
                            verification_mode='no_checks')

    en_pre_train_ds = MT6PreTrainingDataset(cc100_en, tok_en, input_max_length=128,
                                            noise_fn=MT5NoiseFunction(noise_density=0.15))
    fr_pre_train_ds = MT6PreTrainingDataset(cc100_fr, tok_en, input_max_length=128,
                                            noise_fn=MT5NoiseFunction(noise_density=0.15))
    de_pre_train_ds = MT6PreTrainingDataset(cc100_de, tok_en, input_max_length=128)

    en_fr_transl_ds = load_dataset("yhavinga/ccmatrix", "en-fr",
                                   cache_dir="/data/n.dallanoce/cc_en_fr",
                                   split=f"train[0:20000000]",
                                   verification_mode='no_checks')
    en_de_transl_ds = load_dataset("yhavinga/ccmatrix", "en-de",
                                   cache_dir="/data/n.dallanoce/cc_en_de",
                                   split=f"train[0:20000000]",
                                   verification_mode='no_checks')

    en_fr_tsc_ds = MT6TranslationDataset(en_fr_transl_ds, tok_en, src_lang="en", tgt_lang="fr", input_max_length=128,
                                         noise_fn=MT5NoiseFunction(noise_density=0.3),
                                         skip_rows={2372581, 6968567, 10821748, 11060884, 15767927, 25424386, 29725453,
                                                    45747545, 47137798, 50129051, 59177023, 59929203, 61511560,
                                                    75542580,
                                                    100970169, 115986518, 127680776, 141459031, 156717917, 157018533,
                                                    162558439, 164150364, 175041176, 184342700, 190148649, 190148650,
                                                    192658445, 220362372, 245452855, 256201123, 271393589, 272871204,
                                                    272877704, 281597372, 294584774, 296244867, 321887045,
                                                    35870050, 48145532, 52684654, 58751416, 58882125, 65601877,
                                                    67930837,
                                                    77241694, 92977227, 110216804, 128101180, 134271264, 141335940,
                                                    163685146, 170148774, 174846035, 175041176, 178472316, 187909576,
                                                    190148650, 190788599, 199867191, 202841440, 203367259, 216538756,
                                                    216971114, 217029239, 217343772, 221404922, 228346708, 229954946,
                                                    236475781, 238254814, 239313560, 240741477, 244758333, 244990634,
                                                    246716684, 246848239, 251633313, 252437626, 258612355, 260023316,
                                                    261848203, 266413071, 269838607, 271039088, 280243425, 290825579,
                                                    303987750, 304028810, 310067703, 310183397, 314725258, 323880921,
                                                    324665884
                                                    })

    en_de_tsc_ds = MT6TranslationDataset(en_de_transl_ds, tok_en, src_lang="en", tgt_lang="de", input_max_length=128,
                                         noise_fn=MT6NoiseFunction(n_groups=3, noise_density=0.5, pnat=True),
                                         skip_rows={2801169, 3015352, 19775415, 20367662, 20785493, 23611708, 25969951,
                                                    32771958, 33590564, 33799669, 38165983, 38349415, 42732422,
                                                    45639868, 46533951, 48154585, 51122630, 52569871, 53253769,
                                                    53605070, 55897441, 56864670, 66495923, 67445650, 72252425,
                                                    72300952, 72964017, 73662839, 76210766, 78842479, 78842480,
                                                    78842481, 82414911, 83162203, 83162204, 83275757, 89471839,
                                                    92184954, 92184955, 105087545, 110218143, 115819739, 123521912,
                                                    130686046, 141563952, 146476551, 147853628,
                                                    154125145, 155297361, 161224397, 175150743, 182672297, 185766813,
                                                    193096473, 202301926, 202497244, 205450756, 208256156, 219628486,
                                                    227123665, 27922110, 31580101, 32771958, 40763567, 45639868,
                                                    63058682, 72964017,
                                                    73662839, 78842480, 79531236, 83162203, 83238534, 92184955,
                                                    105831028,
                                                    111265110, 145839309, 164197978, 185766813, 188391705, 198131661,
                                                    198435949, 199964247, 230948265, 240673484, 245361151, 246664703})

    train_ds = ConcatDataset([en_pre_train_ds, fr_pre_train_ds, en_fr_tsc_ds])
    trainer = MT6Trainer("pretraining", model, training_args,
                         train_dataset=train_ds,
                         # optimizers=(optimizer, lr_scheduler)
                         )
    trainer.train(resume_from_checkpoint=False)
