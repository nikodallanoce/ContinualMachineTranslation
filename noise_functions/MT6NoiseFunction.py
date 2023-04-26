from typing import List, Tuple, Optional, Union
import numpy as np


class MT6NoiseFunction:

    def __init__(self, n_groups: int = 3, noise_density: float = 0.5, return_list=True, span_length: int = 3):
        self.n_groups: int = n_groups
        self.noise_density: float = noise_density
        self.return_list: bool = return_list
        self.span_length = span_length

    def compute(self, text: str, seed: int) -> Tuple[
        str, Union[List[str], str]]:

        src_tokens: List[str] = list(filter(None, text.split(" ")))
        src_tokens, trg_tokens, n_span = self.mask_src_trg_span_length(self.noise_density, seed, src_tokens)
        # trg_tokens, n_span = self.mask_src_trg(self.noise_density, seed, src_tokens)

        span_per_group, span_reminder = divmod(n_span, self.n_groups)
        targets: Union[List[str], str] = []
        sent = ""
        counter = 0
        for i in range(len(trg_tokens)):
            token = trg_tokens[i]
            sent = sent + " " + token

            if i == 0:
                sent = token

            if "<extra_id_" in token and i > 0:
                counter += 1

            if (counter == (span_per_group + int(span_reminder > 0))) or i == (len(trg_tokens) - 1):
                targets.append(sent)
                sent = token
                counter = 0
                span_reminder -= 1

        if not self.return_list:
            targets = " ".join(targets)
        return " ".join(filter(None, src_tokens)), targets

    @staticmethod
    def index_inside_bounds(index: int, span_len: int, bounds: List[Tuple[int, int]]):
        for b in bounds:
            if b[0] <= index + span_len and index <= b[1]:
                return True
        return False

    def compute_span_indexes(self, noise_density: float, seed: int, src_tokens: List[str]):
        rng = np.random.default_rng(seed)
        span_bounds_idxs: List[Tuple[int, int]] = []
        tokens_to_mask = round(len(src_tokens) * noise_density)
        # span_length, start_mask_idx = self.generate_index_and_span_len(rng, src_tokens)
        # span_bounds_idxs.append((start_mask_idx, start_mask_idx + span_length))
        # tokens_to_mask = tokens_to_mask - span_length
        patience: int = 0
        while tokens_to_mask > 0 and patience < 100:
            patience = 0
            span_length, start_mask_idx = self.generate_index_and_span_len(rng, src_tokens)
            while (self.index_inside_bounds(start_mask_idx, span_length,
                                            span_bounds_idxs) or span_length == 0) and patience < 100:
                span_length, start_mask_idx = self.generate_index_and_span_len(rng, src_tokens)
                if span_length > tokens_to_mask:
                    span_length = tokens_to_mask
                patience = patience + 1
            tokens_to_mask = tokens_to_mask - span_length
            span_bounds_idxs.append((start_mask_idx, start_mask_idx + span_length))
            #patience = 0
        span_bounds_idxs.sort(key=lambda x: x[0])
        return span_bounds_idxs

    def generate_index_and_span_len(self, rng: np.random.Generator, src_tokens: List[str]):
        span_length = rng.poisson(self.span_length)
        if len(src_tokens) - span_length < 2:
            span_length = 0
        start_mask_idx = rng.integers(1, len(src_tokens) - span_length, endpoint=False)
        return span_length, start_mask_idx

    def mask_src_trg_span_length(self, noise_density: float, seed: int, src_tokens: List[str]):
        span_bounds_idx: List[Tuple[int, int]] = self.compute_span_indexes(noise_density, seed, src_tokens)
        src_tokens: np.ndarray = np.array(src_tokens, dtype=object)
        src_mask_gen = MaskTokenGenerator()
        tgt_mask_gen = MaskTokenGenerator()
        tgt_tokens: List[str] = [next(tgt_mask_gen)]
        n_span = 0
        for bounds in span_bounds_idx:
            i_s, i_e = bounds
            masked_src = list(src_tokens[i_s:i_e])
            src_tokens[i_s:i_e] = ""
            src_tokens[i_s] = next(src_mask_gen)
            tgt_tokens.extend(masked_src)
            if i_e < len(src_tokens):
                tgt_tokens.append(next(tgt_mask_gen))
                n_span += 1

        # src, tgt = " ".join(filter(None, src_tokens)), " ".join(filter(None, tgt_tokens))

        return src_tokens, tgt_tokens, len(span_bounds_idx)

    def mask_src_trg(self, noise_density: float, seed: int, src_tokens: List[str]) -> Tuple[List[str], int]:
        rng = np.random.default_rng(seed)
        trg_tokens: List[str] = []
        mask_tok_src = MaskTokenGenerator()
        mask_tok_trg = MaskTokenGenerator()
        src_masked: bool = False
        trg_masked: bool = False
        n_span: int = 0
        for i in range(len(src_tokens)):
            curr_tok = src_tokens[i]
            if rng.random() < noise_density:
                trg_masked = False
                trg_tokens.append(curr_tok)
                if not src_masked:
                    src_tokens[i] = next(mask_tok_src)
                    src_masked = True
                    n_span += 1
                else:
                    src_tokens[i] = ""
            else:
                src_masked = False
                if not trg_masked:
                    trg_tokens.append(next(mask_tok_trg))
                    trg_masked = True
        return trg_tokens, n_span


class MaskTokenGenerator:
    def __init__(self):
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        mask_tok = f"<extra_id_{self.i}>"
        self.i += 1
        return mask_tok


if __name__ == '__main__':
    import sys

    sys.path.insert(0, '/home/n.dallanoce/PyCharm/pretraining')
    from transformers import MT5TokenizerFast
    from datasets import load_dataset
    from custom_datasets.MT6PreTrainingDataset import MT6PreTrainingDataset
    from torch.utils.data import DataLoader
    from utilities.utility import collate_pad
    from functools import partial
    from tqdm import tqdm

    # pre_train_ds = load_dataset("cc100", lang="en",
    #                             cache_dir="/data/n.dallanoce/cc100/huggingface",
    #                             split=f"train[{4096}:{4096 * 2}]",
    #                             verification_mode='no_checks')
    # pre_train_ds = load_dataset("cc100", lang="en",
    #                             cache_dir="/data/n.dallanoce/cc100/huggingface",
    #                             split=f"train[0:40000000]",
    #                             verification_mode='no_checks')

    tok_en = MT5TokenizerFast.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5")
    #ds = MT6PreTrainingDataset(pre_train_ds, tok_en)
    original = "We introduce how to convert the following three types of the language understanding task into the text-to-text format. Under this setting, the models should be fine-tuned only on English training data but evaluated on all target languages. Moreover, for each pretrained model, only one model is used for all languages rather than selecting fine-tuned models separately."
    src, trg = MT6NoiseFunction(return_list=True, noise_density=0.5).compute(text=original, seed=85)
    # for e in tqdm(DataLoader(ds, batch_size=128, collate_fn=partial(collate_pad, pad_token_id=tok_en.pad_token_id),
    #                          num_workers=16)):
    #     pass
    # tokenized = tok_en(trg, add_special_tokens=True, max_length=16, padding="max_length", truncation=True)
    # lst = src.split()
    # masked_w = sum(1 for x in lst if "<extra_id" in x)
    # new_len = len(lst) - masked_w
    # mask_prc = 1 - (new_len / len(original.split()))
    print()
