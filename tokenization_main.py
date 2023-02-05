from typing import List, Dict

from datasets import concatenate_datasets, load_dataset, load_from_disk, DownloadMode
from torch.utils.data import DataLoader
from transformers import MBartTokenizer


def tokenize(examples: List[Dict[str, str]], **kwargs):
    tokenizer: MBartTokenizer = kwargs['tokenizer']
    lang: str = kwargs['lang']
    batch_src: List[str] = [e['en'] for e in examples]
    # tokenize the batch of sentences
    tokenized_src = tokenizer(batch_src, return_special_tokens_mask=False, truncation=True,
                                 max_length=tokenizer.model_max_length // 2, padding='max_length', return_tensors='pt')
    batch_src: List[str] = [e[lang] for e in examples]
    tokenized_trg = tokenizer(batch_src, return_special_tokens_mask=False, truncation=True,
                                 max_length=tokenizer.model_max_length // 2, padding='max_length', return_tensors='pt')

    return {'src_ids': tokenized_src.data['input_ids'], 'trg_ids': tokenized_trg.data['input_ids']}


if __name__ == '__main__':
    lang_pair = "en-de"
    cache_dir = "/data/n.dallanoce/cc_" + lang_pair.replace("-", "_")
    dataset = load_dataset("yhavinga/ccmatrix", lang_pair, split='train', cache_dir=cache_dir,
                           ignore_verifications=True, streaming=True)

    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")
    #dataset = dataset["translation"]
    dataset = dataset.map(tokenize, batched=True, input_columns=['translation'],
                          fn_kwargs={'tokenizer': tokenizer, 'lang': 'de'})
    dataset = dataset.remove_columns(['id', 'score', 'translation'])
    dataset = dataset.with_format('torch')

    dataset = DataLoader(dataset, batch_size=2, drop_last=True)
    for e in iter(dataset):
        en = e['translation']['en'][0]
        print()

print()
