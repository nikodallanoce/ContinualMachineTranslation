import datasets
from tokenizers.processors import ByteLevel
from tokenizers.implementations import SentencePieceBPETokenizer
from transformers import MBartTokenizer, MBartTokenizerFast, AutoTokenizer, MT5TokenizerFast
from datasets import load_dataset, concatenate_datasets, Translation, Features
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler


def monolingual_corpus(length: int):
    if length == -1:
        length = "100%"
    en_ds = load_dataset("cc100", lang="en",
                         cache_dir="/data/n.dallanoce/cc100/huggingface",
                         split=f"train[0:{length}]",
                         verification_mode='no_checks')
    es_ds = load_dataset("cc100", lang="es",
                         cache_dir="/data/n.dallanoce/cc100/huggingface",
                         split=f"train[0:{length}]",
                         verification_mode='no_checks')
    de_ds = load_dataset("cc100", lang="de",
                         cache_dir="/data/n.dallanoce/cc100/huggingface",
                         split=f"train[0:{length}]",
                         verification_mode='no_checks')
    fr_ds = load_dataset("cc100", lang="fr",
                         cache_dir="/data/n.dallanoce/cc100/huggingface",
                         split=f"train[0:{length}]",
                         verification_mode='no_checks')
    return en_ds, es_ds, de_ds, fr_ds


def collate(batch):
    list_sent = []
    for record in batch:
        for lang, sent in record['translation'].items():
            list_sent.append(sent)

    return list_sent


def map_hugg(batch):
    list_dent = []
    for elem in batch:
        for key, val in elem.items():
            list_dent.append(val)
    return {'sent': list_dent}


def get_training_corpus(raw_dataset: datasets.Dataset, field: str, batch_dim: int):
    for start_idx in tqdm(range(0, len(raw_dataset), batch_dim)):
        end_idx = start_idx + batch_dim
        end_idx = end_idx if end_idx < len(raw_dataset) else len(raw_dataset) - 1
        yield raw_dataset[start_idx:end_idx][field]
        # sentences = list(map(lambda x: x.strip(), samples[field]))
        # yield sentences


def tok_gen(dataloader):
    for e in tqdm(dataloader):
        yield e


def file_generator(file):
    with open(file, mode="r", encoding="UTF-8") as text:
        yield text.readline().strip()


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("t5-base") #facebook/mbart-large-cc25

    # outp = tokenizer(src, text_target=tgt, return_special_tokens_mask=False,
    #                  add_special_tokens=True, truncation=True,
    #                  max_length=5,
    #                  return_tensors='pt')
    batch_dim = 10000
    # tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M", mask_token="<mask>", cls_token="<length>")

    # list_ds = list(monolingual_corpus(-1))
    # min_len = min([len(ds) for ds in list_ds])
    min_len = 379272696
    min_len = min_len // 5
    print(min_len)
    list_ds = list(monolingual_corpus(min_len))
    dataset = concatenate_datasets(list_ds)
    # #dataset = dataset.shuffle()
    #
    #
    # sentencepiece_tokenizer = SentencePieceBPETokenizer()
    # sentencepiece_tokenizer.post_processor = ByteLevel()
    # special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<length>", "<mask>"]
    # sentencepiece_tokenizer.train_from_iterator(
    #     get_training_corpus(dataset, field="text", batch_dim=batch_dim), vocab_size=48000,
    #     special_tokens=special_tokens)
    # sentencepiece_tokenizer.save("/home/n.dallanoce/PyCharm/pretraining/tokenizers/sentencepiece_vocab_48k.json")

    # tokenizer = MBartTokenizerFast.from_pretrained("facebook/mbart-large-cc25")
    new_tokenizer = tokenizer.train_new_from_iterator(get_training_corpus(dataset, field="text", batch_dim=batch_dim),
                                                      vocab_size=32000)
    new_tokenizer.save_pretrained("/home/n.dallanoce/PyCharm/pretraining/tokenizers/mt5_cc4_32k_5")
    new_tokenizer.push_to_hub("mt5-cc4-vanilla-32k-5")
    # small_tokenizer.save_pretrained("/home/n.dallanoce/PyCharm/pretraining/tokenizers/mbart")
    # small_tokenizer.save_pretrained("/home/n.dallanoce/PyCharm/pretraining/tokenizers/mbart_en")
