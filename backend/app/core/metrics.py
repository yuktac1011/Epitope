import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, roc_curve

def calculate_threshold_from_validation(y_val, y_val_probs, target_fpr=0.01):
    fpr, tpr, thresholds = roc_curve(y_val, y_val_probs)
    valid_fpr_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_fpr_indices) > 0:
        return thresholds[valid_fpr_indices[-1]]
    return thresholds[-1]

def evaluate_metrics(y_true, y_probs, threshold):
    ap = average_precision_score(y_true, y_probs)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    y_pred = (y_probs >= threshold).astype(int)
    
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    actual_positives = np.sum(y_true == 1)
    recall = float(true_positives / actual_positives) if actual_positives > 0 else 0.0
    
    return {
        "ap": float(ap),
        "pr_auc": float(pr_auc),
        "recall_at_target_fpr": recall
    }

def aggregate_macro_metrics(bank_metrics: dict):
    if not bank_metrics:
        return {}
    keys = list(bank_metrics.values())[0].keys()
    macro = {}
    for k in keys:
        macro[k] = float(np.mean([bm[k] for bm in bank_metrics.values()]))
    return macro
