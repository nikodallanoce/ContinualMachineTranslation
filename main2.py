import torch
from transformers import MBartTokenizer, MBartConfig
from datasets import load_from_disk
from torch.optim import *
from torch.utils.data import DataLoader
from models.MBart import MBart
from MBartDataset import MBartDataset


def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb


if __name__ == '__main__':
    # torch.backends.cudnn.benchmark = True
    print(torch.cuda.is_available())

    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")

    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=128, decoder_ffn_dim=128,
                               encoder_attention_heads=4, decoder_attention_heads=4,
                               d_model=256, max_length=128, vocab_size=tokenizer.vocab_size)

    model: MBart = MBart(mbart_config)
    print(model_size(model))

    dataset_loaded = load_from_disk("europarl_eng_tokenized")
    # concat_ds = concatenate_datasets([dataset_loaded, dataset_loaded])

    my_ds = MBartDataset(dataset_loaded, tokenizer, 1)
    ds_en_loader = DataLoader(my_ds, batch_size=32, drop_last=True, shuffle=True, pin_memory=True, num_workers=16)
    model.fit(ds_en_loader, Adam(model.parameters()), steps=5)
