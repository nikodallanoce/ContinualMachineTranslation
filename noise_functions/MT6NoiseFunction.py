from typing import List, Tuple, Optional, Union

from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import numpy as np


class MT6NoiseFunction:

    def __init__(self, n_groups: int = 3, noise_density: float = 0.5, return_list=False):
        self.n_groups: int = n_groups
        self.noise_density: float = noise_density
        self.return_list: bool = return_list

    def compute(self, text: str, seed: int) -> Tuple[
        str, Union[List[str], str]]:

        src_tokens: List[str] = list(filter(None, text.split(" ")))
        trg_tokens, n_span = self.mask_src_trg(self.noise_density, seed, src_tokens)

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

    def compute_for_mt5(self, text: str, seed: int, noise_density: float = 0.15) -> Tuple[
        str, str]:

        src_tokens: List[str] = list(filter(None, text.split(" ")))
        trg_tokens, n_span = self.mask_src_trg(noise_density, seed, src_tokens)
        return " ".join(filter(None, src_tokens)), " ".join(filter(None, trg_tokens))

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
            if rng.random() <= noise_density:
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
    tok_en = MT5Tokenizer.from_pretrained("google/mt5-base")
    original = "We introduce how to convert the following three types of the language understanding task into the text-to-text format."
    src, trg = MT6NoiseFunction().compute_for_mt5(
        original,
        seed=1, noise_density=0.35)
    # tokenized = tok_en(trg, add_special_tokens=True, max_length=16, padding="max_length", truncation=True)
    lst = src.split()
    masked_w = sum(1 for x in lst if "<extra_id" in x)
    new_len = len(lst) - masked_w
    mask_prc = 1 - (new_len / len(original.split()))
    print()
