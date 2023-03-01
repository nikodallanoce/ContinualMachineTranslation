from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MBartTokenizer, MBartConfig

from models.MBart import MBart
from noise_functions.MBartNoiseFunction import MBartNoiseFunction
from custom_datasets.MBartPreTrainingDataset import MBartPreTrainingDataset


def testing():
    pre_train_load = DataLoader(pre_train_ds, batch_size=16, drop_last=True, shuffle=False)
    for e in tqdm(pre_train_load):
        pass


def train():
    model: MBart = MBart(mbart_config, device_ids=[3])
    #model.load_state_dict(torch.load("mbart_v2_test.pt"))
    # optimizer = Adam(model.parameters(), eps=1e-6:, betas=(0.98, 0.999))
    optimizer = Adam(model.parameters())
    # lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_training_steps=5e5, num_warmup_steps=0)
    pre_train_load = DataLoader(pre_train_ds, batch_size=16, drop_last=True, shuffle=True, pin_memory=True,
                                num_workers=8)
    model.fit_epoch(pre_train_load, optimizer, epochs=50)


if __name__ == '__main__':

    noise_fn = MBartNoiseFunction()
    #sent = "I must say that, while I won't recommend this novel to anyone, I think it was powerful. It felt to me like performance art, something I endured, an artistic experience which divides people but is undeniably compelling."
    #noise_fn.compute(sent)

    # translation_ds = load_dataset("text", data_files={"train": ["/data/n.dallanoce/cc100/en.txt"]},
    #                             cache_dir="/data/n.dallanoce/cc100/hugg_en", split=f"train[{300000}:{360000}]",
    #                             ignore_verifications=True)

    pre_train_ds = load_dataset("text", data_files={"train": ["test_hugg_en/test_data_hugg.txt"]},
                                cache_dir="test_hugg_en", split='train[:100%]',
                                ignore_verifications=True)

    #noise_fn.compute("Hello point. And she is. How is him? Sorry for that.")

    tok_en = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")
    t = tok_en("Hello sir... . I buy on amazon.com.")
    tokens = tok_en.batch_decode(t.input_ids)

    pre_train_ds = MBartPreTrainingDataset(pre_train_ds, tok_en, "en_XX")
    #
    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                               encoder_attention_heads=8, decoder_attention_heads=8,
                               d_model=512, max_length=128, vocab_size=tok_en.vocab_size)
    testing()
    #train()
