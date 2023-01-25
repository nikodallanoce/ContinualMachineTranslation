from datasets import concatenate_datasets, load_dataset, load_from_disk, DownloadMode


if __name__ == '__main__':
    dataset = load_dataset("yhavinga/ccmatrix", "en-fr", split='train', cache_dir="/data/n.dallanoce/cc_en_fr",
                           download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS, ignore_verifications=True)