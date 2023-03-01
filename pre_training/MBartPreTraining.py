from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import MBartTokenizer, MBartConfig, get_scheduler
from accelerate import Accelerator
from models.MBart import MBart

import sys

sys.path.insert(0, '../custom_datasets/')
from custom_datasets.MBartPreTrainingDataset import MBartPreTrainingDataset

if __name__ == '__main__':
    pre_train_ds = load_dataset("text", data_files={"train": ["/data/n.dallanoce/cc100/en.txt"]},
                                cache_dir="/data/n.dallanoce/cc100/hugg_en", split='train', ignore_verifications=True)

    tok_en = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")

    pre_train_ds = MBartPreTrainingDataset(pre_train_ds, tok_en, "en")

    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                               encoder_attention_heads=8, decoder_attention_heads=8,
                               d_model=512, max_length=256, vocab_size=tok_en.vocab_size)

    accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=1)
    model: MBart = MBart(mbart_config, device_ids=[3])
    optimizer = Adam(model.parameters(), eps=1e-6, betas=(0.98, 0.999))
    # optimizer = Adam(model.parameters())
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_training_steps=5e5, num_warmup_steps=0)

    pre_train_load = DataLoader(pre_train_ds, batch_size=16, drop_last=True, shuffle=False, pin_memory=True,
                                num_workers=2)

    # model, optimizer, pre_train_load, lr_scheduler = accelerator.prepare(model, optimizer , pre_train_load, lr_scheduler)
    # optimizer = accelerator.prepare_optimizer(optimizer, device_placement='cuda')

    model.fit(pre_train_load, optimizer, steps=500000, lr_scheduler=lr_scheduler)

    # model.fit(load_en, Adam(model.parameters()), epochs=1)

    # for e in pre_train_load:
    #    print()
