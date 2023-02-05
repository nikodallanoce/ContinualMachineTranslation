from datasets import concatenate_datasets, load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import MBartTokenizer, MBartConfig, MBartForConditionalGeneration

from MBart import MBart
from MBartDataset import MBartDataset

if __name__ == '__main__':

    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50", "/data/n.dallanoce/mbart-large-50")

    cc_en_de = load_dataset("yhavinga/ccmatrix", "en-de", split='train', cache_dir="/data/n.dallanoce/cc_en_de",
                            ignore_verifications=True)

    tok_en = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")
    tok_de = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="de_DE")
    tok_es = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="es_XX")
    tok_fr = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="fr_XX")

    data_en = MBartDataset(cc_en_de, tok_en, "en")
    data_de = MBartDataset(cc_en_de, tok_de, "de")

    load_en = DataLoader(data_en, batch_size=8, drop_last=True, shuffle=True, pin_memory=True, num_workers=16)

    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                               encoder_attention_heads=8, decoder_attention_heads=8,
                               d_model=512, max_length=256, vocab_size=tok_en.vocab_size)

    model: MBart = MBart(mbart_config, device_ids=[3])

    e = next(iter(load_en))
    print()

    model.fit(load_en, Adam(model.parameters()), steps=1)
