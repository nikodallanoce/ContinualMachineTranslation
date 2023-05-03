from datasets import load_dataset

if __name__ == '__main__':

    val_ds_es_en = load_dataset("nikodallanoce/wmt14", "es-en",
                                split=f"validation",
                                streaming=True, use_auth_token=True)

    test_ds_es_en = load_dataset("nikodallanoce/wmt14", "es-en",
                                 split=f"test",
                                 streaming=True, use_auth_token=True)

    for val_sent, test_sent in zip(iter(val_ds_es_en), iter(test_ds_es_en)):
        print((val_sent, test_sent))
        print(val_sent == test_sent)
