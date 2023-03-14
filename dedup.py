import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm


def compute_collisions(dataset, i_s, thr):
    # dataset = dataset[i_s:i_e]
    collision_indexes: List[int] = []
    ds = dataset['translation']
    #eta = tqdm(ds, desc=f"{thr}")
    #eta.set_postfix(thread=thr)
    for i, translation in enumerate(ds):
        sent = translation['en']
        if hash(sent) in string_hashes:
            if sent == string_hashes[hash(sent)]:
                collision_indexes.append(i_s + i)
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
    test_ds = load_dataset("wmt14", "fr-en",
                           cache_dir="/data/n.dallanoce/wmt14",
                           split=f"test",
                           ignore_verifications=True)
    string_hashes: Dict[int, str] = dict()

    sent: str
    for translation in test_ds:
        sent = translation['translation']['en']
        string_hashes[hash(sent)] = sent

    train_ds = load_dataset("yhavinga/ccmatrix", "en-fr",
                            cache_dir="/data/n.dallanoce/cc_en_fr",
                            split=f"train",
                            ignore_verifications=True)

    # train_ds = load_dataset("wmt14", "fr-en",
    #                         cache_dir="D:\\datasets\\wmt14",
    #                         split=f"test",
    #                         ignore_verifications=True)

    num_of_threads = 2**18
    # executor = ThreadPoolExecutor(num_of_threads)
    sent_per_thread, reminder = divmod(len(train_ds), num_of_threads)

    i_s, i_e = 0, sent_per_thread
    results: List[Future] = []

    with ThreadPoolExecutor(num_of_threads) as executor:
        for thr in tqdm(range(num_of_threads)):
            results.append(executor.submit(compute_collisions, train_ds[i_s:i_e], i_s, thr))
            i_s, i_e, reminder = start_end_indexes(i_s, i_e, reminder)

        duplicates_idxs = []
        for res in results:
            duplicates_idxs.extend(res.result())

    print(len(duplicates_idxs))
