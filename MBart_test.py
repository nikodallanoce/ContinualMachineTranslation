from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MBartTokenizer, MBartConfig, MBartForConditionalGeneration

from custom_datasets.MBartTranslationDataset import MBartTranslationDataset


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
    translation_ds = load_dataset("wmt14", "fr-en",
                                  cache_dir="D:\\datasets\\wmt14",
                                  split=f"train[:1024]",
                                  ignore_verifications=True)
    # translation_ds = load_dataset("text", data_files={"train": ["test_hugg_en/test_data_hugg.txt"]},
    #                             cache_dir="test_hugg_en", split='train[:64]',
    #                             ignore_verifications=True)
    # translation_ds = load_dataset("g8a9/europarl_en-it",
    #                             cache_dir="/data/n.dallanoce/europarl", split=f"train",
    #                             ignore_verifications=True)

    tok_en = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="fr_XX")

    translation_ds = MBartTranslationDataset(translation_ds, tok_en, "fr")

    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                               encoder_attention_heads=8, decoder_attention_heads=8,
                               d_model=512, max_length=128, vocab_size=tok_en.vocab_size)
    model: MBartForConditionalGeneration = MBartForConditionalGeneration(mbart_config).to('cuda:0')

    for batch in tqdm(DataLoader(translation_ds)):
        input_ids, att_mask, label_ids = batch['input_ids'], batch['attention_mask'], batch['labels']
        label_ids = label_ids.to('cuda:0')
        att_mask = att_mask.to('cuda:0')
        input_ids = input_ids.to('cuda:0')

        loss = model(input_ids, att_mask, labels=label_ids)
#
