from typing import Tuple
from transformers import EvalPrediction
import evaluate
from evaluate import Metric


def get_metric():
    return evaluate.load("accuracy")


def get_compute_metrics_function(metric: Metric):
    def compute_metrics(result: EvalPrediction):
        predictions = result.predictions[0] if isinstance(result.predictions, Tuple) else result.predictions
        label_ids = result.label_ids
        output = metric.compute(predictions=predictions, references=label_ids)
        return output

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, Tuple):
        logits = logits[0]
    predictions = logits.argmax(dim=-1)
    return predictions
