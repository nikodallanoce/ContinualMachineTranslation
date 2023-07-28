from functools import partial

import math
from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM
from statistics import mean

from utilities.utility import collate_pad


def compute_pretraining_metrics(model: PreTrainedModel, dataset: Dataset, batch_size: int = 32,
                                device: str = "cuda:0") -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    tmp_ris: Dict[str, torch.Tensor] = next(iter(dataset))
    lab_name = "labels"
    for k in tmp_ris.keys():
        if k.startswith("label"):
            lab_name = k
    dl = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=2,
                    collate_fn=partial(collate_pad, pad_token_id=model.config.pad_token_id, labels_name=lab_name))
    mean_loss = 0
    for batch in tqdm(dl):
        if "labels" not in batch.keys():
            batch['labels'] = batch[lab_name]
            del batch[lab_name]
        for k in batch:
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            loss = model(**batch).loss
            mean_loss += loss.item()

    mean_loss = mean_loss / len(dl)
    perplexity = math.exp(mean_loss)
    metrics["loss"] = mean_loss
    metrics["perplexity"] = perplexity
    return metrics
