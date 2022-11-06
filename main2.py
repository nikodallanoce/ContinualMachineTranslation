import torch
from torch import nn
from transformers import MBartModel, MBartTokenizer, MBartConfig
from transformers import AutoTokenizer
import multiprocessing
from datasets import concatenate_datasets, load_dataset, load_from_disk
from torch.optim import *
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
from CustomDataset import CustomDataset
from MBart import MBart


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
    print(torch.cuda.is_available())

    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")

    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=512, decoder_ffn_dim=512,
                               encoder_attention_heads=8, decoder_attention_heads=8,
                               d_model=512, max_length=128, vocab_size=tokenizer.vocab_size)

    model: MBart = MBart(mbart_config)
    print(model_size(model))

    dataset_loaded = load_from_disk("europarl_eng_tokenized_and_masked_128")

    dataset_loaded.set_format(type='pt', columns=['input_ids', 'attention_mask', 'masked_ids'])
    dataset_loaded = dataset_loaded[1:100]

    input_ids = dataset_loaded['input_ids']
    attention_mask = dataset_loaded['attention_mask']
    masked_ids = dataset_loaded['masked_ids']

    ds_en_loader = DataLoader(CustomDataset(masked_ids, input_ids, attention_mask),
                              batch_size=4, drop_last=True, shuffle=True,
                              pin_memory=True, pin_memory_device='cuda', num_workers=4)

    model.fit(ds_en_loader, Adam(model.parameters()), epochs=10)
