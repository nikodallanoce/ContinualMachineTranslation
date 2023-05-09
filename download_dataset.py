from datasets import concatenate_datasets, load_dataset, load_from_disk, DownloadMode
import datasets
from tqdm import tqdm

from noise_functions.MT5NoiseFunction import MT5NoiseFunction

if __name__ == '__main__':
    en_mc4 = load_dataset("mc4", languages=["en"], split="train", streaming=True)
    fr_mc4 = load_dataset("mc4", languages=["fr"], split="train", streaming=True)
    en_fr_mc4 = datasets.interleave_datasets([en_mc4, fr_mc4])

    noise_fn = MT5NoiseFunction()
    sep = "--"
    for i in range(128):
        sep += "--"
    i = 0
    with open("masked_t5_file.txt", mode="w+", encoding="UTF-8") as write_file:
        for elem in tqdm(iter(en_fr_mc4), total=1000):
            txt_split = elem['text'].split()
            text = elem['text'] if len(txt_split) < 128 else " ".join(txt_split[0:128])
            src_txt, tgt_txt = noise_fn.compute(text, i)
            write_file.write(f"text: {text} \n\n")
            write_file.write(f"src: {src_txt} \n\n")
            write_file.write(f"tgt: {tgt_txt} \n")
            write_file.write(sep + "\n")
            # print()
            i = i + 1
            if i == 1000:
                break

    print()
