from datasets import concatenate_datasets, load_dataset, load_from_disk, DownloadMode

if __name__ == '__main__':
    dataset = load_dataset("yhavinga/ccmatrix", "en-de", split='train', cache_dir="/data/n.dallanoce/cc_en_de",
                           ignore_verifications=True)

    print()
