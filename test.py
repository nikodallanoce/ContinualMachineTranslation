import random
from typing import List, Tuple, Dict

import numpy as np


def span_already_masked(span_start_idx: int, span_length: int, spans_dict: Dict[int, List[str]]):
    for i in range(span_start_idx, span_start_idx + span_length):
        if i in spans_dict:
            return True
    return False


def index_inside_bounds(index: int, span_len: int, bounds: List[Tuple[int, int]]):
    for b in bounds:
        if b[0] <= index + span_len and index <= b[1]:
            return True
    return False


def compute_span_indexes(noise_density: float, seed: int, src_tokens: List[str]):
    rng = np.random.default_rng(seed)
    span_bounds_idxs: List[Tuple[int, int]] = []
    tokens_to_mask = round(len(src_tokens) * noise_density)
    span_length = rng.poisson(3)
    start_mask_idx = rng.integers(1, len(src_tokens) - span_length)
    span_bounds_idxs.append((start_mask_idx, start_mask_idx + span_length))
    tokens_to_mask = tokens_to_mask - span_length
    while tokens_to_mask > 0:
        span_length = rng.poisson(3)
        start_mask_idx = rng.integers(1, len(src_tokens) - span_length)
        while index_inside_bounds(start_mask_idx, span_length, span_bounds_idxs):
            span_length = rng.poisson(3)
            start_mask_idx = rng.integers(1, len(src_tokens) - span_length)
        tokens_to_mask = tokens_to_mask - span_length
        span_bounds_idxs.append((start_mask_idx, start_mask_idx + span_length))
    span_bounds_idxs.sort(key=lambda x: x[0])
    return span_bounds_idxs


class MaskTokenGenerator:
    def __init__(self):
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        mask_tok = f"<extra_id_{self.i}>"
        self.i += 1
        return mask_tok


def mask(noise_density: float, seed: int, src_tokens: List[str]):
    span_bounds_idx: List[Tuple[int, int]] = compute_span_indexes(noise_density, seed, src_tokens)
    src_tokens: np.ndarray = np.array(src_tokens, dtype=object)
    src_mask_gen = MaskTokenGenerator()
    tgt_mask_gen = MaskTokenGenerator()
    tgt_tokens: List[str] = [next(tgt_mask_gen)]
    for bounds in span_bounds_idx:
        i_s, i_e = bounds
        masked_src = list(src_tokens[i_s:i_e])
        src_tokens[i_s:i_e] = ""
        src_tokens[i_s] = next(src_mask_gen)
        tgt_tokens.extend(masked_src)
        if i_e < len(src_tokens):
            tgt_tokens.append(next(tgt_mask_gen))

    src, tgt = " ".join(filter(None, src_tokens)), " ".join(filter(None, tgt_tokens))

    return src, tgt


def remove_words(noise_density: float, seed: int, src_tokens: List[str]):
    rng = np.random.default_rng(seed)
    targets: Dict[int, List[str]] = {}
    num_words = len(src_tokens)
    num_remove = int(num_words * noise_density)
    start = 1
    while num_remove > 0:
        span_length = rng.poisson(3)
        start = random.randint(0, num_words - span_length)
        while span_already_masked(start, span_length, targets):
            start = random.randint(0, num_words - span_length)
        src_tok_to_remove = src_tokens[start:start + 3]
        targets[start] = src_tok_to_remove
        src_tokens[start + (span_length - 1)] = "<mask>"
        src_tokens[start:start + (span_length - 1)] = ""
        num_words -= span_length
    return ' '.join(src_tokens)


if __name__ == '__main__':
    sentence = "We introduce how to convert the following three types of the language understanding task into the text-to-text format. Under this setting, the models should be fine-tuned only on English training data but evaluated on all target languages. Moreover, for each pretrained model, only one model is used for all languages rather than selecting fine-tuned models separately."
    print(mask(noise_density=0.5, seed=85, src_tokens=sentence.split()))
