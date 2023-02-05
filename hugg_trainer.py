from datasets import load_dataset
from transformers import Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments, MBartTokenizer, MBartConfig, \
    MBartForConditionalGeneration
from transformers.integrations import TensorBoardCallback

from MBart import MBart
from MBartPreTrainingDataset import MBartPreTrainingDataset

if __name__ == '__main__':
    pre_train_ds = load_dataset("text", data_files={"train": ["/data/n.dallanoce/cc100/en.txt"]},
                                cache_dir="/data/n.dallanoce/cc100/hugg_en", split='train[:1024]',
                                ignore_verifications=True)
    tok_en = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")

    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                               encoder_attention_heads=8, decoder_attention_heads=8,
                               d_model=512, max_length=128, vocab_size=tok_en.vocab_size)
    model: MBartForConditionalGeneration = MBartForConditionalGeneration(mbart_config)
    pre_train_ds = MBartPreTrainingDataset(pre_train_ds, tok_en, "en_XX")

    training_args = Seq2SeqTrainingArguments("/home/n.dallanoce/PyCharm/pretraining/hugg_trainer/mbart/",
                                             overwrite_output_dir=True,
                                             do_train=True,
                                             per_device_train_batch_size=16,
                                             num_train_epochs=2,
                                             max_steps=-1,
                                             log_level="debug",
                                             save_strategy="epoch",
                                             fp16=True,
                                             dataloader_drop_last=True,
                                             dataloader_pin_memory=True,
                                             dataloader_num_workers=8,
                                             # prediction_loss_only=True,
                                             save_total_limit=3,
                                             metric_for_best_model="train_loss",
                                             greater_is_better=False,
                                             report_to=["tensorboard"]
                                             )
    trainer = Seq2SeqTrainer(model, training_args,
                             train_dataset=pre_train_ds,
                             callbacks=[TensorBoardCallback()])
    trainer.train()
