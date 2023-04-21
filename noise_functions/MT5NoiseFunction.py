from typing import Tuple, List

from noise_functions.MT6NoiseFunction import MT6NoiseFunction


class MT5NoiseFunction(MT6NoiseFunction):
    def __init__(self, noise_density: float = 0.5, span_length: int = 3):
        return_list: bool = False
        n_groups: int = 1
        super().__init__(n_groups, noise_density, return_list, span_length)

    def compute(self, text: str, seed: int) -> Tuple[str, str]:
        src_tokens: List[str] = list(filter(None, text.split(" ")))
        src_tokens, tgt_tokens, n_span = self.mask_src_trg_span_length(self.noise_density, seed, src_tokens)
        return " ".join(filter(None, src_tokens)), " ".join(filter(None, tgt_tokens))


if __name__ == '__main__':
    # from transformers import MT5TokenizerFast
    #
    # tok_en = MT5TokenizerFast.from_pretrained("nikodallanoce/mt5-cc4-vanilla-32k-5")
    original = "We introduce how to convert the following three types of the language understanding task into the text-to-text format. Under this setting, the models should be fine-tuned only on English training data but evaluated on all target languages. Moreover, for each pretrained model, only one model is used for all languages rather than selecting fine-tuned models separately."
    src, trg = MT5NoiseFunction(noise_density=0.5).compute(text=original, seed=85)
    # tokenized = tok_en(trg, add_special_tokens=True, max_length=16, padding="max_length", truncation=True)
    # lst = src.split()
    # masked_w = sum(1 for x in lst if "<extra_id" in x)
    # new_len = len(lst) - masked_w
    # mask_prc = 1 - (new_len / len(original.split()))
    print()
