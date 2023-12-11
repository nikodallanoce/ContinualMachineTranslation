import time
from functools import partial

import datasets
import torch.nn
from datasets import load_dataset
import transformers
from torch.optim import AdamW, Adam, SGD
from torch.utils.data import ConcatDataset
from transformers import Seq2SeqTrainingArguments, MT5ForConditionalGeneration, MT5Config, MT5TokenizerFast, \
    T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, \
    get_cosine_with_hard_restarts_schedule_with_warmup, Adafactor, get_inverse_sqrt_schedule, \
    get_linear_schedule_with_warmup
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

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
project_name = "mt5_ft_en-fr_M1F1"
os.environ["WANDB_PROJECT"] = project_name

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "false"

# turn off watch to log faster

os.environ["WANDB_WATCH"] = "false"
# os.environ["WANDB_DISABLED"] = "true"

if __name__ == '__main__':

    training_args = Seq2SeqTrainingArguments(
        f"/data/n.dallanoce/weights/{project_name}_t5_mbart_lion",
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
        logging_steps=250,
        save_steps=5000,
        save_strategy="steps",
        log_level="info",
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=5000,
        logging_first_step=False,
        fp16=True,
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        # prediction_loss_only=True,
        save_total_limit=1,
        metric_for_best_model="bleu_avg",
        greater_is_better=True,
        report_to=["wandb"],
        ignore_data_skip=False,
        save_safetensors=False,
        torch_compile=torch.__version__.startswith("2")
    )

    # model = MT5ForConditionalGeneration(
    #     MT5Config(num_layers=6, num_decoder_layers=6, d_model=512, num_heads=8, d_ff=2048, vocab_size=len(tok),
    #               max_length=133, tie_word_embeddings=True))
    # model = T5ForConditionalGeneration.from_pretrained(
    #     "google/t5-v1_1-small", max_length=max_inp_len, tie_word_embeddings=False)
    # model = MT5ForConditionalGeneration(
    #     MT5Config(num_layers=6, d_model=512, num_heads=8, d_ff=2048,
    #               vocab_size=len(AutoTokenizer.from_pretrained("google/t5-v1_1-small")), max_length=133,
    #               tie_word_embeddings=True))
    # time.sleep(12600)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path="/data/n.dallanoce/weights/mt5_pre_en-fr(M1)_t5_mbart/checkpoint-180000",
        # pretrained_model_name_or_path="google/t5-v1_1-small",
    )

    train_strategy = TrainingStrategy.FINE_TUNING

    if train_strategy == TrainingStrategy.FINE_TUNING:
        choice: str = input(
            "Do you want to use T5 original tokenizer, the custom one or the automatic one?\nAnswer t5, custom or auto: ")
        if choice == "t5":
            tok_name = "google/t5-v1_1-small"
        elif choice == "custom":
            tok_name = "nikodallanoce/mt5-cc4-vanilla-32k-5"
        elif choice == "auto":
            tok_name = "nikodallanoce/mt5-cc4-vanilla-32k-5"
            if not model.name_or_path.startswith("/"):
                tok_name = model.name_or_path
        else:
            raise ValueError("Invalid choice.")
        tok = AutoTokenizer.from_pretrained(tok_name)
        max_inp_len = 133
    elif train_strategy == TrainingStrategy.FINE_TUNING_LANG:
        tok_name = "nikodallanoce/mt6_tok_fast"
        tok = MT6TokenizerFast.from_pretrained(tok_name)
        max_inp_len = 128
        print("Using lang tokens")
    elif not model.name_or_path.startswith("/"):
        tok_name = model.name_or_path
        tok = AutoTokenizer.from_pretrained(tok_name)
        max_inp_len = 133
    else:
        raise ValueError("Select FINE TUNING or FINE TUNING LANG")

    translation_ds_en_fr = load_dataset("yhavinga/ccmatrix", "en-fr",
                                        cache_dir="/data/n.dallanoce/ccmatrix",
                                        split=f"train[0:35000000]",
                                        verification_mode='no_checks')

    en_fr_ds = MT6TranslationDataset(translation_ds_en_fr, tok, src_lang="en", tgt_lang="fr",
                                     input_max_length=max_inp_len,
                                     noise_fn=None,
                                     skip_rows={2372581, 6968567, 10821748, 11060884, 15767927, 25424386, 29725453,
                                                45747545, 47137798, 50129051, 59177023, 59929203, 61511560, 75542580,
                                                100970169, 115986518, 127680776, 141459031, 156717917, 157018533,
                                                162558439, 164150364, 175041176, 184342700, 190148649, 190148650,
                                                192658445, 220362372, 245452855, 256201123, 271393589, 272871204,
                                                272877704, 281597372, 294584774, 296244867, 321887045},
                                     concat_sents=True)
    fr_en_ds = MT6TranslationDataset(translation_ds_en_fr, tok, src_lang="fr", tgt_lang="en",
                                     input_max_length=max_inp_len,
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
                                                324665884}, concat_sents=True)
    translation_ds_en_de = load_dataset("yhavinga/ccmatrix", "en-de",
                                        cache_dir="/data/n.dallanoce/ccmatrix",
                                        split=f"train[0:35000000]",
                                        verification_mode='no_checks')

    en_de_ds = MT6TranslationDataset(translation_ds_en_de, tok, src_lang="en", tgt_lang="de", noise_fn=None,
                                     input_max_length=max_inp_len,
                                     skip_rows={2801169, 3015352, 19775415, 20367662, 20785493, 23611708, 25969951,
                                                32771958, 33590564, 33799669, 38165983, 38349415, 42732422, 45639868,
                                                46533951, 48154585, 51122630, 52569871, 53253769, 53605070, 55897441,
                                                56864670, 66495923, 67445650, 72252425, 72300952, 72964017, 73662839,
                                                76210766, 78842479, 78842480, 78842481, 82414911, 83162203, 83162204,
                                                83275757, 89471839, 92184954, 92184955, 105087545, 110218143,
                                                115819739, 123521912, 130686046, 141563952, 146476551, 147853628,
                                                154125145, 155297361, 161224397, 175150743, 182672297, 185766813,
                                                193096473, 202301926, 202497244, 205450756, 208256156, 219628486,
                                                227123665})
    de_en_ds = MT6TranslationDataset(translation_ds_en_de, tok, src_lang="de", tgt_lang="en", noise_fn=None,
                                     input_max_length=max_inp_len,
                                     skip_rows={27922110, 31580101, 32771958, 40763567, 45639868, 63058682, 72964017,
                                                73662839, 78842480, 79531236, 83162203, 83238534, 92184955, 105831028,
                                                111265110, 145839309, 164197978, 185766813, 188391705, 198131661,
                                                198435949, 199964247, 230948265, 240673484, 245361151, 246664703})

    translation_ds_en_es = load_dataset("yhavinga/ccmatrix", "en-es",
                                        cache_dir="/data/n.dallanoce/ccmatrix",
                                        split=f"train[0:35000000]",
                                        verification_mode='no_checks')

    en_es_ds = MT6TranslationDataset(translation_ds_en_es, tok, src_lang="en", tgt_lang="es",
                                     input_max_length=max_inp_len, noise_fn=None,
                                     skip_rows={104162920, 201875059, 220054035, 239139109, 242048881, 250889199,
                                                299111881, 300507236, 357223245})
    es_en_ds = MT6TranslationDataset(translation_ds_en_es, tok, src_lang="es", tgt_lang="en",
                                     input_max_length=max_inp_len, noise_fn=None,
                                     skip_rows={6751585, 7725692, 35698791, 43267448, 48075653, 48925282, 48925283,
                                                53971646, 55073237, 56061226, 59861190, 63114112, 80366657, 80902620,
                                                86295984, 87272571, 111795284, 113916617, 118570344, 119519063,
                                                120600774, 129587251, 130850605, 136272300, 136519043, 136733349,
                                                137378072, 137491231, 137665670, 143721844, 145405838, 146284528,
                                                146621153, 153890220, 155856369, 159312479, 160583224, 163400236,
                                                167062018, 167230158, 168709714, 171236351, 173926019, 175953391,
                                                180093066, 181892264, 188671101, 190817212, 191197133, 191440526,
                                                193089123, 193231113, 202625957, 218523240, 219695892, 225112141,
                                                227156035, 231918342, 233796239, 233865889, 237259151, 274202045,
                                                278337171, 288838952, 289266873, 290450151, 291301513, 293807105,
                                                294314852, 296624934, 297053788, 303558652, 317079342, 318616807,
                                                322993811, 326540772, 329582867, 341969381, 349023337, 352305431,
                                                354125230, 356161656, 357435039, 364171762, 378343556, 380637241,
                                                393342927, 398415139, 399539622, 404908528, 407859230})

    val_ds_fr_en = load_dataset("wmt14", "fr-en",
                                cache_dir="/data/n.dallanoce/wmt14",
                                split=f"validation",
                                verification_mode='no_checks')

    val_ds_de_en = load_dataset("wmt14", "de-en",
                                cache_dir="/data/n.dallanoce/wmt14",
                                split=f"validation",
                                verification_mode='no_checks')

    val_ds_name = "nikodallanoce/wmt14"
    # val_ds_es_en = load_dataset(val_ds_name, "es-en",
    #                             cache_dir="/data/n.dallanoce/wmt14",
    #                             split=f"validation",
    #                             verification_mode='no_checks', use_auth_token=True)
    val_ds_es_en = load_dataset("yhavinga/ccmatrix", "en-es",
                                cache_dir="/data/n.dallanoce/ccmatrix",
                                split=f"train[40000000:40003000]",
                                verification_mode='no_checks').with_format("torch", columns=['translation'])
    val_ds_config_en_fr = val_ds_fr_en.config_name.replace("-", "_")  # works with wmt14 datasets
    val_ds_config_en_de = val_ds_de_en.config_name.replace("-", "_")
    val_ds_config_en_es = val_ds_es_en.config_name.replace("-", "_")

    # if train_strategy == TrainingStrategy.FINE_TUNING_LANG:
    #     model.resize_token_embeddings(len(tok))
    # time.sleep(5*60*60)
    from lion_pytorch.lion_pytorch import Lion

    optim = Lion(filter(lambda p: p.requires_grad, model.parameters()), lr=training_args.learning_rate)

    trainer = MT6Trainer(train_strategy, model, training_args,
                         train_dataset=ConcatDataset([en_fr_ds, fr_en_ds]),
                         eval_dataset={"bleu": ConcatDataset([val_ds_fr_en])},
                         tokenizer_name=tok.name_or_path,
                         optimizers=(
                             optim, get_linear_schedule_with_warmup(optim, num_warmup_steps=0,
                                                                    num_training_steps=training_args.max_steps))
                         )

    trainer.train(resume_from_checkpoint=False)
