import torch
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, MBartTokenizer, MBartConfig, \
    MBartForConditionalGeneration, Seq2SeqTrainer
import sys

sys.path.insert(0, '/home/n.dallanoce/PyCharm/pretraining')
from custom_datasets.MBartTranslationDataset import MBartTranslationDataset
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == '__main__':
    size = str(int(2 ** 24))
    # translation_ds = load_dataset("wmt14", "fr-en",
    #                               cache_dir="/data/n.dallanoce/wmt14",
    #                               split=f"train",
    #                               ignore_verifications=True)
    # translation_ds = load_dataset("wmt14", "fr-en",
    #                               cache_dir="D:\\datasets\\wmt14",
    #                               split=f"train[0:2048]",
    #                               ignore_verifications=True)
    translation_ds = load_dataset("yhavinga/ccmatrix", "en-fr",
                                  cache_dir="/data/n.dallanoce/cc_en_fr",
                                  split=f"train[0:20000000]",
                                  ignore_verifications=True)

    tok_en = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="fr_XX")

    translation_ds = MBartTranslationDataset(translation_ds, tok_en, "fr")

    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                               encoder_attention_heads=8, decoder_attention_heads=8,
                               d_model=512, max_length=128, vocab_size=tok_en.vocab_size, dropout=0.3)
    model: MBartForConditionalGeneration = MBartForConditionalGeneration(mbart_config)

    model.load_state_dict(
        torch.load(
            "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_cc100/checkpoint-499800/pytorch_model.bin",
            map_location='cuda:0'))
    training_args = Seq2SeqTrainingArguments("/home/n.dallanoce/PyCharm/pretraining/weights/mbart_ft_fr-en_cc_ls/",
                                             overwrite_output_dir=True,
                                             label_names=['labels'],
                                             do_train=True,
                                             label_smoothing_factor=0.1,
                                             warmup_steps=2500,
                                             optim="adamw_torch",
                                             #learning_rate=3e-5,
                                             #auto_find_batch_size=True,
                                             per_device_train_batch_size=16,
                                             gradient_accumulation_steps=1,
                                             #num_train_epochs=60,
                                             max_steps=int(5e5),
                                             logging_steps=1000,
                                             save_steps=10000,
                                             log_level="info",
                                             save_strategy="steps",
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
    # training_args = Seq2SeqTrainingArguments("D:\\trainer\\mbart_en_fr",
    #                                          overwrite_output_dir=True,
    #                                          label_names=['labels'],
    #                                          do_train=True,
    #                                          label_smoothing_factor=0.2,
    #                                          warmup_steps=2500,
    #                                          optim="adamw_torch",
    #                                          learning_rate=3e-5,
    #                                          #auto_find_batch_size=True,
    #                                          per_device_train_batch_size=4,
    #                                          gradient_accumulation_steps=1,
    #                                          num_train_epochs=40,
    #                                          #max_steps=int(5e5),
    #                                          logging_steps=100,
    #                                          save_steps=10000,
    #                                          log_level="info",
    #                                          save_strategy="steps",
    #                                          fp16=True,
    #                                          dataloader_drop_last=True,
    #                                          dataloader_pin_memory=True,
    #                                          dataloader_num_workers=8,
    #                                          # prediction_loss_only=True,
    #                                          save_total_limit=2,
    #                                          metric_for_best_model="loss",
    #                                          greater_is_better=False,
    #                                          report_to=["tensorboard"]
    #                                          )
    # optimizer = Adam(model.parameters(), eps=1e-6, betas=(0.9, 0.98))
    # optimizer = Adam(model.parameters())
    # lr_scheduler = transformers.get_constant_schedule(optimizer)
    # lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_training_steps=43740, num_warmup_steps=0)
    # lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_training_steps=500000, num_warmup_steps=0)
    trainer = Seq2SeqTrainer(model, training_args,
                             train_dataset=translation_ds,
                             # optimizers=(optimizer, lr_scheduler)
                             )
    trainer.train(resume_from_checkpoint=False)
