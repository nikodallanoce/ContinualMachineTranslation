from transformers import MBartTokenizer, MBartTokenizerFast
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


def get_training_corpus(raw_dataset, field: str):
    batch_dim = 10000
    for start_idx in tqdm(range(0, len(raw_dataset), batch_dim)):
        samples = raw_dataset[start_idx: start_idx + batch_dim]
        yield samples[field]


if __name__ == '__main__':
    tokenizer = MBartTokenizerFast.from_pretrained("facebook/mbart-large-cc25")
    dataset = load_dataset("cc100", lang="en",
                           cache_dir="/data/n.dallanoce/cc100/huggingface",
                           split=f"train",
                           verification_mode='no_checks')

    train_corpus = get_training_corpus(dataset, "text")
    small_tokenizer = tokenizer.train_new_from_iterator(train_corpus, 52000)
    small_tokenizer.save_pretrained("/home/n.dallanoce/PyCharm/pretraining/tokenizers/mbart")
