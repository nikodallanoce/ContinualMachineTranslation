from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, MBartTokenizer, MBartConfig, \
    MBartForConditionalGeneration

import sys
sys.path.insert(0, '/home/n.dallanoce/PyCharm/pretraining')
from custom_datasets.MBartPreTrainingDataset import MBartPreTrainingDataset
from trainers.MBartTrainer import MBartTrainer

if __name__ == '__main__':
    size = str(int(2 ** 24))
    # pre_train_ds = load_dataset("cc100", lang="en",
    #                             cache_dir="/data/n.dallanoce/cc100/hugg_en",
    #                             split=f"train[{0}:{10000000}]",
    #                             ignore_verifications=True)
    pre_train_ds = load_dataset("text", data_files={"train": ["D:\\datasets\\test_hugg_en\\test_data_hugg.txt"]},
                                cache_dir="D:\\datasets\\test_hugg_en", split=f'train[0:128]',
                                ignore_verifications=True)
    # translation_ds = load_dataset("g8a9/europarl_en-it",
    #                             cache_dir="/data/n.dallanoce/europarl", split=f"train",
    #                             ignore_verifications=True)

    tok_en = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")

    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                               encoder_attention_heads=8, decoder_attention_heads=8,
                               d_model=512, max_length=128, vocab_size=tok_en.vocab_size)
    model: MBartForConditionalGeneration = MBartForConditionalGeneration(mbart_config)
    pre_train_ds = MBartPreTrainingDataset(pre_train_ds, tok_en, "en_XX")


    # training_args = Seq2SeqTrainingArguments("/home/n.dallanoce/PyCharm/pretraining/weights/mbart_cc100/",
    #                                          overwrite_output_dir=True,
    #                                          label_names=['labels'],
    #                                          do_train=True,
    #                                          #auto_find_batch_size=True,
    #                                          per_device_train_batch_size=16,
    #                                          gradient_accumulation_steps=1,
    #                                          num_train_epochs=1,
    #                                          max_steps=int(5e5),
    #                                          logging_steps=100,
    #                                          save_steps=300,
    #                                          log_level="info",
    #                                          save_strategy="steps",
    #                                          load_best_model_at_end= True
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
    training_args = Seq2SeqTrainingArguments("D:\\trainer\\mbart",
                                             overwrite_output_dir=True,
                                             label_names=['labels'],
                                             do_train=True,
                                             per_device_train_batch_size=4,
                                             num_train_epochs=10,
                                             max_steps=-1,
                                             log_level="debug",
                                             save_strategy="epoch",
                                             fp16=True,
                                             dataloader_drop_last=True,
                                             dataloader_pin_memory=True,
                                             dataloader_num_workers=1,
                                             # prediction_loss_only=True,
                                             save_total_limit=1,
                                             metric_for_best_model="loss",
                                             greater_is_better=False,
                                             report_to=["tensorboard"]
                                             )
    # optimizer = Adam(model.parameters(), eps=1e-6, betas=(0.9, 0.98))
    # optimizer = Adam(model.parameters())
    # lr_scheduler = transformers.get_constant_schedule(optimizer)
    # lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_training_steps=43740, num_warmup_steps=0)
    # lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_training_steps=500000, num_warmup_steps=0)
    trainer = MBartTrainer(model, training_args,
                           train_dataset=pre_train_ds,
                           # optimizers=(optimizer, lr_scheduler)
                           )
    trainer.train(resume_from_checkpoint=False)
