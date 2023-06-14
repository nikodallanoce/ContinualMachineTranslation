from typing import Union, Optional, Dict, Callable, List, Tuple, Any

import torch
from torch import nn
from torch.utils.data import Dataset, ConcatDataset
from transformers import Seq2SeqTrainer, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback

from eval.bleu_utility import compute_bleu_auto_model


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, model: Union[PreTrainedModel, nn.Module] = None, args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None, train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Optional[Callable[[], PreTrainedModel]] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)

    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval", **gen_kwargs) -> Dict[str, float]:
        from statistics import mean
        eval_metrics: Dict[str, float] = dict()
        if isinstance(eval_dataset, ConcatDataset):
            for eval_ds in eval_dataset.datasets:
                langs = eval_ds.features['translation'].languages
                src_lang, tgt_lang = langs[0], langs[1]
                metrics: Dict[str, float] = self.update_metrics_dict(eval_ds, src_lang, tgt_lang)
                eval_metrics.update(metrics)
        else:
            langs = eval_dataset.features['translation'].languages
            src_lang, tgt_lang = langs[0], langs[1]
            metrics: Dict[str, float] = self.update_metrics_dict(eval_dataset, src_lang, tgt_lang)
            eval_metrics.update(metrics)

        eval_metrics['eval_bleu_avg'] = mean(eval_metrics[k] for k in eval_metrics)
        return_metrics = eval_metrics.copy()
        self.log(eval_metrics)
        return return_metrics

    def update_metrics_dict(self, eval_ds: Dataset, src_lang, tgt_lang) -> Dict[str, float]:
        metrics: Dict[str, Any] = dict()
        for i in range(2):
            bleu_score: Dict[str, Any] = compute_bleu_auto_model(eval_ds, self.model,
                                                                 src_lang=src_lang,
                                                                 tgt_lang=tgt_lang,
                                                                 bleu_type="sacrebleu",
                                                                 batch_size=32)
            metric_key = next(iter(bleu_score))
            src_lang, tgt_lang = tgt_lang, src_lang
            metrics[f"eval_bleu_{src_lang}_{tgt_lang}"] = bleu_score[metric_key]
        return metrics
