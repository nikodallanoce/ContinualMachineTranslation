import random
from typing import List, Tuple

import numpy as np


class MBartNoiseFunction:

    def __init__(self, mask_percentage: float = 0.35, eos_sep: str = "</s>", mask_token: str = "<mask>",
                 lang: str = "en_XX", poisson_span_len: float = 3.5):
        self.mask_percentage: float = mask_percentage
        self.eos_sep: str = eos_sep
        self.mask_token: str = mask_token
        self.lang = lang
        self.poisson_span_len = poisson_span_len

    def compute(self, sentence: str, seed: int = 0) -> Tuple[str, str]:
        sentence = sentence.strip()
        rng = np.random.default_rng(seed)
        permuted_sent: np.ndarray = np.array([sent.strip() for sent in filter(None, sentence.split(self.eos_sep))],
                                             dtype=object)
        rng.shuffle(permuted_sent)

        for i in range(0, len(permuted_sent)):
            sent = permuted_sent[i]
            sent = np.array(sent.split(" "), dtype=object)
            self.span_masking(sent, seed)
            permuted_sent[i] = " ".join(filter(None, sent)).rstrip()
            if i < len(permuted_sent) - 1:
                permuted_sent[i] += self.eos_sep

        # permuted_sent = (self.eos_sep + " ").join(permuted_sent).rstrip() + self.eos_sep

        # words: np.ndarray = np.array(permuted_sent.split(" "), dtype=np.str)

        masked_words = " ".join(filter(None, permuted_sent))  # + " " + self.eos_sep + " " + self.lang
        # lst = masked_words.split()
        # masked_w = lst.count("<mask>")
        # new_len = len(lst) - masked_w
        # mask_prc = 1 - (new_len / len(sentence.split()))
        return sentence, masked_words

    def span_masking(self, words: np.ndarray, seed: int):
        tokens_length = len(words)
        num_tokens_to_mask = round(tokens_length * self.mask_percentage)
        rng = np.random.default_rng(seed)
        m_end = 0
        while num_tokens_to_mask > 0:
            mask_span_len = rng.poisson(self.poisson_span_len)
            if mask_span_len > num_tokens_to_mask:
                mask_span_len = num_tokens_to_mask
            index = rng.integers(m_end, tokens_length - num_tokens_to_mask, dtype=int)

            if index != m_end or index == 0:
                words[index] = self.mask_token
            else:
                words[index] = ""

            m_end = mask_span_len + index
            words[index + 1:m_end] = ""

            # m_end = m_end + 1
            # num_tokens_to_mask -= 1
            num_tokens_to_mask = num_tokens_to_mask - mask_span_len

    def insert_eos(self, words: np.ndarray, indexes_of_eos: List[int]) -> List[str]:
        new_sent: List[str] = list(words)

        for i in indexes_of_eos:
            new_sent.insert(i, self.eos_sep)
        return new_sent

    @staticmethod
    def get_indexes_eos_token(words: np.ndarray) -> List[int]:
        indexes: List[int] = []
        i: int = 0
        while i < len(words):
            w: str = words[i]
            if w.endswith(".") or w.endswith("?") or w.endswith("!") or i == (words.size - 1):
                indexes.append(i + 1)
            i = i + 1
        return indexes


if __name__ == '__main__':
    MBartNoiseFunction(0.35).compute(
        "As additional baselines, we will also include a comparison.</s> With a model randomly initialized without pre-training.</s> Process finished with exit code -1",
        6)
