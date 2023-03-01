from datasets import concatenate_datasets, load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import MBartTokenizer, MBartConfig, MBartForConditionalGeneration

from MBart import MBart
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
    tok_en = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")

    # translation_ds = MBartPreTrainingDataset(translation_ds, tok_en, "en")

    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                               encoder_attention_heads=8, decoder_attention_heads=8,
                               d_model=512, max_length=128, vocab_size=tok_en.vocab_size)

    # accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=1)
    # model: MBart = MBart(mbart_config, device_ids=[3])
    cuda_dev = "cpu"
    # model = model.to(cuda_dev)
    model = MBartForConditionalGeneration(mbart_config).to(cuda_dev)
    model.train(False)
    print(model_size(model))

    model_size_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Transformer-base, number of trainable parameters: {0}".format(model_size_trainable))
