import random
from typing import List, Tuple

import numpy as np


class MBartNoiseFunction:

    def __init__(self, mask_percentage: float = 0.35, eos_sep: str = "</s>", mask_token: str = "<mask>",
                 lang: str = "en_XX"):
        self.mask_percentage: float = mask_percentage
        self.eos_sep: str = eos_sep
        self.mask_token: str = mask_token
        self.lang = lang
        self.full_stop = "."

    # def compute(self, sentence: str, seed: int = 0) -> Tuple[str, str]:
    #     sentence = sentence.strip()
    #     words: np.ndarray = np.array(sentence.split(" "), dtype=np.str)
    #     indexes_of_eos: List[int] = self.get_indexes_eos_token(words)
    #     indexes_of_eos.reverse()
    #
    #     sentence = self.insert_eos(words.copy(), indexes_of_eos)
    #     sentence = " ".join(sentence) + " " + self.lang
    #
    #     self.span_masking(words, seed)
    #
    #     masked_words = self.insert_eos(words, indexes_of_eos)
    #
    #     #p = " ".join(filter(None, masked_words))
    #     sentence_list = []
    #     sent = ""
    #     for w in masked_words:
    #         if w != "":
    #             sent += w
    #             if w.endswith(self.eos_sep):
    #                 sentence_list.append(sent)
    #                 sent = ""
    #             else:
    #                 sent += " "
    #     random.seed(seed)
    #     random.shuffle(sentence_list)
    #     masked_words = " ".join(sentence_list) + " " + self.lang
    #     return sentence, masked_words

    def compute(self, sentence: str, seed: int = 0) -> Tuple[str, str]:
        sentence = sentence.strip()
        rng = np.random.default_rng(seed)
        permuted_sent = [sent.strip() for sent in filter(None, sentence.split(self.full_stop))]
        rng.shuffle(permuted_sent)

        permuted_sent = (self.full_stop + " ").join(permuted_sent).rstrip() + self.full_stop

        words: np.ndarray = np.array(permuted_sent.split(" "), dtype=np.str)

        # sentence += " " + self.eos_sep + " " + self.lang

        self.span_masking(words, seed)

        masked_words = " ".join(filter(None, words))  # + " " + self.eos_sep + " " + self.lang
        return sentence, masked_words

    def span_masking(self, words: np.ndarray, seed: int):
        tokens_length = len(words)
        num_tokens_to_mask = round(tokens_length * self.mask_percentage)
        rng = np.random.default_rng(seed)
        m_end = 0
        while num_tokens_to_mask > 0 and m_end < (tokens_length - num_tokens_to_mask):
            mask_span_len = rng.poisson(3.5)
            if mask_span_len > num_tokens_to_mask:
                mask_span_len = num_tokens_to_mask
            index = np.random.randint(m_end, tokens_length - num_tokens_to_mask)
            m_end = mask_span_len + index

            words[index] = self.mask_token
            words[index + 1:m_end] = ""

            m_end = m_end + 1
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
