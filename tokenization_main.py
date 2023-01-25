from typing import List, Dict

from datasets import concatenate_datasets, load_dataset, load_from_disk, DownloadMode
from torch.utils.data import DataLoader
from transformers import MBartTokenizer


def tokenize(examples: List[Dict[str, str]], **kwargs):
    tokenizer: MBartTokenizer = kwargs['tokenizer']
    lang: str = kwargs['lang']
    inp: List[str] = [e[lang] for e in examples]
    # tokenize the batch of sentences
    tokenized_inputs = tokenizer(inp, return_special_tokens_mask=False, truncation=True,
                                 max_length=tokenizer.model_max_length // 2, padding='max_length', return_tensors='pt')

    return {'input_ids': tokenized_inputs.data['input_ids']}


if __name__ == '__main__':
    dataset = load_dataset("yhavinga/ccmatrix", "en-es", split='train', streaming=True, ignore_verifications=True)

    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")

    dataset = dataset.map(tokenize, batched=True, input_columns=['translation'],
                          fn_kwargs={'tokenizer': tokenizer, 'lang': 'en'})
    dataset = dataset.with_format('torch')
    dataset = DataLoader(dataset, batch_size=2, drop_last=True)
    for e in iter(dataset):
        # en = e['translation']['en'][0]
        print()

    print()
