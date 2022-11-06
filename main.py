import torch
from torch import nn
from transformers import MBartModel, MBartTokenizer, MBartConfig
from transformers import AutoTokenizer
import multiprocessing
from datasets import concatenate_datasets, load_dataset, load_from_disk
from tqdm import tqdm
from torch.utils import data
from MBart import MBart
from random import random
import numpy as np


def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb


def batch_iterator(dataset, batch_size=10000):
    for i in tqdm(range(0, len(dataset), batch_size)):
        yield dataset[i: i + batch_size]


def tokenize_and_mask(examples: list[str], **kwargs):
    tokenizer: MBartTokenizer = kwargs['tokenizer']
    mask_percentage = kwargs['mask_percentage']

    # tokenize the batch of sentences
    tokenized_inputs = tokenizer(examples, return_special_tokens_mask=False, truncation=True,
                                 max_length=tokenizer.model_max_length // 8, padding='max_length', return_tensors='np')

    tokenized_ids = np.copy(tokenized_inputs.data['input_ids'])
    masked_ids = mask_tokens(tokenized_ids, mask_percentage, tokenizer)

    return {'input_ids': tokenized_inputs.data['input_ids'],
            'attention_mask': tokenized_inputs.data['attention_mask'],
            'masked_ids': masked_ids}


def mask_tokens(tokenized_ids, mask_percentage, tokenizer):
    # replace some token that differ from the special ones with a mask token
    for sentence_ids in tokenized_ids:
        # sentence:np.ndarray = sentence_ids
        for j in range(sentence_ids.shape[0]):
            if sentence_ids[j] == 1:
                break
            if sentence_ids[j] not in tokenizer.all_special_ids:
                if random() <= mask_percentage:  # and sentence_ids[j - 1] != tokenizer.mask_token_id:
                    sentence_ids[j] = tokenizer.mask_token_id
    return tokenized_ids


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(torch.cuda.is_available())

    mbart_config = MBartConfig(encoder_layers=6, decoder_layers=6,
                               encoder_ffn_dim=512, decoder_ffn_dim=512,
                               encoder_attention_heads=8, decoder_attention_heads=8,
                               d_model=512, max_position_embeddings=2048, max_length=512)

    # model: nn.Module = MBart(mbart_config)

    # print('model size: {:.3f}MB'.format(model_size(model)))

    dataset_europarl = load_dataset("g8a9/europarl_en-it", split='train')
    dataset_europarl = dataset_europarl.remove_columns('Unnamed: 0')

    tokenizer = MBartTokenizer.from_pretrained("nikodallanoce/mbart-europarl-en-it", src_lang="it_IT")
    tok_ris = tokenizer("Ciao amico.", return_special_tokens_mask=True, max_length=20,
                        padding='max_length', add_special_tokens=True)

    num_proc = 1  # multiprocessing.cpu_count()
    print(f"The max length for the tokenizer is: {tokenizer.model_max_length}")

    # preprocess dataset
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")
    tok_en = dataset_europarl.map(tokenize_and_mask, batched=True, num_proc=8, batch_size=10000,
                                  input_columns=['sent_en'], fn_kwargs={'tokenizer': tokenizer, 'mask_percentage': 0.3})

    tok_en = tok_en.remove_columns(['sent_en', 'sent_it'])
    tok_en.save_to_disk("europarl_eng_tokenized_and_masked")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
