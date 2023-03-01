import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, T5Config
from tqdm import tqdm
from custom_datasets.MT6PreTrainingDataset import MT6PreTrainingDataset


def compute_loss(model: MT5ForConditionalGeneration, data):
    input_ids = data["input_ids"]
    input_ids = input_ids.view(1, -1)
    targets = data["labels"]
    targets = targets.view(targets.shape[1], -1)
    loss = 0
    loss += model(input_ids=input_ids, attention_mask=torch.where(input_ids != 0, 1, 0),
                  labels=targets.view(1, -1)).loss


if __name__ == '__main__':
    pre_train_ds = load_dataset("text", data_files={"train": ["test_hugg_en/test_data_hugg.txt"]},
                                cache_dir="test_hugg_en/", split='train',
                                ignore_verifications=True)

    # translation_ds = load_dataset("text", data_files={"train": ["/data/n.dallanoce/cc100/en.txt"]},
    #                             cache_dir="/data/n.dallanoce/cc100/hugg_en", split=f"train[:50%]",
    #                             ignore_verifications=True)

    tok_en = MT5Tokenizer.from_pretrained("google/mt5-base")
    model = MT5ForConditionalGeneration(
        T5Config(vocab_size=tok_en.vocab_size, decoder_start_token_id=tok_en.pad_token_id))

    data_l = DataLoader(MT6PreTrainingDataset(pre_train_ds, tok_en), batch_size=1, drop_last=True)

    for e in tqdm(data_l):
        compute_loss(model, e)
