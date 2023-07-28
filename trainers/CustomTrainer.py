from abc import abstractmethod, ABC
from typing import Union, Optional, Dict, Callable, List, Tuple, Any

import torch
from torch import nn
from torch.utils.data import Dataset, ConcatDataset
from transformers import Seq2SeqTrainer, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback

from custom_datasets.MBartPreTrainingDataset import MBartPreTrainingDataset
from custom_datasets.MT6PreTrainingDataset import MT6PreTrainingDataset
from custom_datasets.MT6TranslationDataset import MT6TranslationDataset
from eval.pretraining_utility import compute_pretraining_metrics


class CustomTrainer(Seq2SeqTrainer, ABC):
    def __init__(self, model: Union[PreTrainedModel, nn.Module] = None, args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None, train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Optional[Callable[[], PreTrainedModel]] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 tokenizer_name: str = None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.tokenizer_name = tokenizer_name

    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval", **gen_kwargs) -> Dict[str, float]:
        from statistics import mean
        self.model.eval()
        eval_metrics: Dict[str, float] = dict()
        is_pretraining: bool = metric_key_prefix.split("_")[1] == "pretraining"
        if not isinstance(eval_dataset, ConcatDataset):
            eval_dataset = ConcatDataset([eval_dataset])
        for eval_ds in eval_dataset.datasets:
            if is_pretraining:
                eval_metrics.update(self.compute_new_pretraining_metrics(eval_ds))
            else:
                eval_metrics.update(self.compute_new_bleu(eval_ds))

        dict_key_avg = metric_key_prefix.split("_")[1]
        if is_pretraining:
            eval_metrics[f'eval_{dict_key_avg}_avg'] = mean(eval_metrics[k] for k in eval_metrics if "loss" in k)
        else:
            eval_metrics[f'eval_{dict_key_avg}_avg'] = mean(eval_metrics[k] for k in eval_metrics)
        return_metrics = eval_metrics.copy()
        self.log(eval_metrics)
        self.model.train()
        return return_metrics

    def compute_new_pretraining_metrics(self, eval_ds: Union[MBartPreTrainingDataset, MT6PreTrainingDataset, MT6TranslationDataset]):
        ris: Dict[str, float] = compute_pretraining_metrics(self.model, eval_ds, batch_size=16, device="cuda:0")
        return_metrics: Dict[str, float] = {}
        for k_r, v_r in ris.items():
            return_metrics[f"eval_pretraining_{k_r}_{eval_ds.dataset.config_name}"] = v_r
        return return_metrics

    def compute_new_bleu(self, eval_ds: Dataset) -> Dict[str, float]:
        langs = eval_ds.features['translation'].languages
        src_lang, tgt_lang = langs[0], langs[1]
        metrics: Dict[str, float] = self.update_metrics_dict(eval_ds, src_lang, tgt_lang)
        return metrics

    def update_metrics_dict(self, eval_ds: Dataset, src_lang, tgt_lang) -> Dict[str, float]:
        metrics: Dict[str, Any] = dict()
        for i in range(2):
            bleu_score: Dict[str, Any] = self.compute_bleu(eval_ds=eval_ds,
                                                           model=self.model,
                                                           src_lang=src_lang,
                                                           tgt_lang=tgt_lang,
                                                           bleu_type="sacrebleu",
                                                           batch_size=32,
                                                           tokenizer_name=self.tokenizer_name)
            metric_key = next(iter(bleu_score))
            metrics[f"eval_bleu_{src_lang}_{tgt_lang}"] = bleu_score[metric_key]
            src_lang, tgt_lang = tgt_lang, src_lang
        return metrics

    @abstractmethod
    def compute_bleu(self, eval_ds: Dataset, model: PreTrainedModel, src_lang: str, tgt_lang: str, bleu_type: str,
                     batch_size: int, tokenizer_name: str) -> Dict[str, Any]:
        raise NotImplementedError("Implement this method on subclasses.")
