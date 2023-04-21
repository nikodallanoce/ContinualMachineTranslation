from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Dict

import datasets
from datasets import load_dataset
from tqdm import tqdm


# def compute_collisions(dataset: datasets.Dataset,# Dict[str, List[Dict[str, str]]],
#                        lang1: str,
#                        i_s: int,
#                        i_e:int,
#                        pbar: tqdm):
#     # dataset = dataset[i_s:i_e]
#     collision_indexes: List[int] = []
#     #ds = dataset.select(range(i_s, i_e, 1))
#     #ds = dataset['translation']
#     # eta = tqdm(ds, desc=f"{thr}")
#     # eta.set_postfix(thread=thr)
#     for i in range(i_s, i_e):
#         translation = dataset[i]['translation']
#         sent = translation[lang1]
#         if hash(sent) in string_hashes:
#             if sent == string_hashes[hash(sent)]:
#                 collision_indexes.append(i_s + i)
#         pbar.update(1)
#     return collision_indexes

def compute_collisions(dataset: Dict[str, List[Dict[str, str]]],
                       lang: str,
                       idx_start: int,
                       pbar_tqdm: tqdm) -> List[int]:
    collision_indexes: List[int] = []
    sentence_pairs = dataset['translation']
    for i, translation in enumerate(sentence_pairs):
        sentence = translation[lang]
        if hash(sentence) in string_hashes and sentence == string_hashes[hash(sentence)]:
            collision_indexes.append(idx_start + i)
        pbar_tqdm.update(1)

    return collision_indexes


def compute_collisions_monolingual(dataset: Dict[str, List[Dict[str, str]]],
                                   feature: str,
                                   idx_start: int,
                                   pbar_tqdm: tqdm,
                                   string_hashes: Dict[int, str]) -> List[int]:
    collision_indexes: List[int] = []
    sentences = dataset[feature]
    for i, sentence in enumerate(sentences):
        if hash(sentence) in string_hashes and sentence == string_hashes[hash(sentence)]:
            collision_indexes.append(idx_start + i)
        pbar_tqdm.update(1)

    return collision_indexes


def start_end_indexes(i_s, i_e, reminder):
    i_s = i_e
    i_e = i_e + sent_per_thread
    if reminder > 0:
        i_e = i_e + 1
        reminder = reminder - 1
    return i_s, i_e, reminder


if __name__ == '__main__':
    # test_ds = load_dataset("wmt14", "fr-en",
    #                        cache_dir="D:\\datasets\\wmt14",
    #                        split=f"test",
    #                        ignore_verifications=True)
    lang1: str = "en"  # final
    lang2: str = "de"
    lang_for_duplicates: str = lang2
    test_ds_name = "wmt14"
    test_ds = load_dataset(test_ds_name, f"{lang2}-{lang1}",
                           cache_dir=f"/data/n.dallanoce/{test_ds_name}",
                           split=f"test",
                           verification_mode='no_checks')
    string_hashes: Dict[int, str] = dict()
    sent: str
    for translation in test_ds:
        sent = translation['translation'][lang_for_duplicates]
        h_s = hash(sent)
        if h_s in string_hashes and sent != string_hashes[h_s]:
            raise Exception("Collision occur")
        string_hashes[h_s] = sent

    # assert len(test_ds) == len(string_hashes)

    train_ds_name = "yhavinga/ccmatrix"
    train_ds = load_dataset(train_ds_name, f"{lang1}-{lang2}",
                            cache_dir=f"/data/n.dallanoce/cc_{lang1}_{lang2}",
                            split=f"train",
                            verification_mode='no_checks')

    # train_ds = load_dataset("cc100", lang1=lang1,
    #                         cache_dir="/data/n.dallanoce/cc100/huggingface",
    #                         split=f"train[0:100%]",
    #                         verification_mode='no_checks')

    # train_ds = load_dataset("wmt14", "fr-en",
    #                         cache_dir="D:\\datasets\\wmt14",
    #                         split=f"train",
    #                         verification_mode='no_checks')
    # train_ds = test_ds

    num_of_threads = 2 ** 18
    # executor = ThreadPoolExecutor(num_of_threads)
    sent_per_thread, reminder = divmod(len(train_ds), num_of_threads)

    i_s, i_e = 0, sent_per_thread
    results: List[Future] = []

    with tqdm(total=len(train_ds)) as pbar:
        with ThreadPoolExecutor(num_of_threads) as executor:
            for thr in tqdm(range(num_of_threads)):
                # tmp_ds = train_ds.select(range(i_s, i_e, 1))
                # results.append(
                #     executor.submit(compute_collisions_monolingual, train_ds[i_s:i_e], "text", i_s, pbar,
                #                     string_hashes))
                results.append(executor.submit(compute_collisions, train_ds[i_s:i_e], lang_for_duplicates, i_s, pbar))
                i_s, i_e, reminder = start_end_indexes(i_s, i_e, reminder)

            duplicates_idxs = []
            for res in results:
                duplicates_idxs.extend(res.result())

    with open(f"dup_idxs_of_ccmatrix_in_{test_ds_name}_{lang_for_duplicates}.txt", mode="w",
              encoding="UTF-8") as out_file:
        ind_to_write = str(duplicates_idxs)
        out_file.write(ind_to_write + "\n")
        out_file.write(f"total number of indexes: {len(duplicates_idxs)}")

    # print(duplicates_idxs)
    print(f"Number of duplicated indexes: {len(duplicates_idxs)}")
