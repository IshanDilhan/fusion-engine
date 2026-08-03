"""Shared classification-metric logging for fusion/rule_based.py and
fusion/gbt.py -- so both scripts report macro- and weighted-averaged
precision/recall/F1 alongside accuracy, not accuracy alone. `labels` should
be the full intent set (not just what's present in this split/subset), so a
class absent from a small split still counts as 0 rather than being dropped
from the average -- consistent across splits of different sizes.
"""
from sklearn.metrics import precision_recall_fscore_support


def log_overall_metrics(mlflow, y_true, y_pred, labels, metric_prefix, print_prefix=None):
    """Logs {metric_prefix}_{macro,weighted}_{precision,recall,f1} to the
    active MLflow run. Returns {"macro": (p, r, f1), "weighted": (p, r, f1)}.
    """
    results = {}
    for avg in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg, zero_division=0)
        mlflow.log_metric(f"{metric_prefix}_{avg}_precision", precision)
        mlflow.log_metric(f"{metric_prefix}_{avg}_recall", recall)
        mlflow.log_metric(f"{metric_prefix}_{avg}_f1", f1)
        results[avg] = (precision, recall, f1)
        if print_prefix:
            print(f"{print_prefix} {avg}: precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
    return results
