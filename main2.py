from datasets import load_dataset
from transformers import AutoTokenizer, MBartTokenizerFast
if __name__ == '__main__':
    tok_name = "nikodallanoce/mbart-cc4-vanilla-32k-5"
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    inp_tok = tokenizer("Hello.</s> How are you?", return_special_tokens_mask=True)
    print()
