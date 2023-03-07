from typing import List, Dict

import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline, MBartConfig, MBartForConditionalGeneration, MBartTokenizer

import os


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def tokenize(examples: List[Dict[str, str]], **kwargs):
    tokenizer: MBartTokenizer = kwargs['tokenizer']
    lang: str = kwargs['lang']
    batch_src: List[str] = [e['en'] for e in examples]
    batch_tgt: List[str] = [e[lang] for e in examples]
    # tokenize the batch of sentences
    outputs = tokenizer(batch_src, return_special_tokens_mask=False,
                        add_special_tokens=True, truncation=True,
                        max_length=128, padding='max_length', return_attention_mask=False,
                        return_tensors='pt')

    return {'input_ids': outputs['input_ids'], 'original_text': batch_tgt}


def collate_tokenize(batch):
    tok_en()

    print()


if __name__ == '__main__':
    tok_en = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="fr_XX")
    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                               encoder_attention_heads=8, decoder_attention_heads=8,
                               d_model=512, max_length=128, vocab_size=tok_en.vocab_size)

    translation_ds = load_dataset("wmt14", "fr-en",
                                  cache_dir="D:\\datasets\\wmt14",
                                  split=f"test[0:2048]",
                                  ignore_verifications=True,
                                  streaming=False)

    tokenized = tok_en("Do you think tomorrow will be a sunny day?", text_target="", add_special_tokens=True,
                       return_tensors='pt')

    translation_ds = translation_ds.map(tokenize, batched=True, input_columns=['translation'],
                                        fn_kwargs={'tokenizer': tok_en, 'lang': 'fr'})
    # translation_ds = translation_ds.remove_columns(['id', 'score', 'translation'])
    translation_ds = translation_ds.with_format('torch')
    test_loader = DataLoader(translation_ds, num_workers=4, batch_size=32, drop_last=True, pin_memory=True)

    model: MBartForConditionalGeneration = MBartForConditionalGeneration(mbart_config).to("cuda:0")
    # model.load_state_dict(
    #     torch.load(
    #         "/home/n.dallanoce/PyCharm/pretraining/weights/mbart_ft_fr-en/checkpoint-500000/pytorch_model.bin",
    #         map_location='cuda:0'))

    model.load_state_dict(
        torch.load("D:\\trainer\\mbart_ft_fr-en\\checkpoint-500000\\pytorch_model.bin", map_location='cuda:0'))

    trans_fr = []
    original_txt = []

    for i, batch in enumerate(tqdm(test_loader)):
        translation = model.generate(batch['input_ids'].to("cuda:0"), max_length=128, decoder_start_token_id=250008,
                                     num_beams=5)
        trans_fr += (tok_en.batch_decode(translation, skip_special_tokens=True))
        original_txt += [[elem] for elem in batch['original_text']]

    # translation = model.generate(tokenized['input_ids'], max_length=128, decoder_start_token_id=250008, num_beams=5)
    # translated_fr = tok_en.batch_decode(translation, skip_special_tokens=True)
    # print(translated_fr)

    bleu = evaluate.load("bleu")
    # decoded_trans = tok_en.batch_decode(trans_fr, skip_special_tokens=True)
    results = bleu.compute(predictions=trans_fr, references=original_txt)
    print(results)

# pipe = pipeline(model=model, task="translation", tokenizer=tok_en, device="cuda:0")
# text_translated = pipe("How are you?", src_lang="en_XX", tgt_lang="fr_XX")
