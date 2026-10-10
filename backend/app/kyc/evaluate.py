import torch
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import numpy as np

def evaluate_model(model, dataloader, device="cpu"):
    """
    Evaluates the model on the provided dataloader.
    Returns honest metrics as requested in the build spec.
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            # Probability for class 1 (Real)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    # Label mapping: 0 = Fake, 1 = Real
    metrics = {}
    if len(np.unique(all_labels)) > 1:
        metrics["ap"] = average_precision_score(all_labels, all_probs)
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
        
        # We consider Fake=0, Real=1.
        # Often in fraud detection, Fake is the "positive" class.
        # But for scikit-learn standard with 1 as positive:
        metrics["precision"] = precision_score(all_labels, all_preds)
        metrics["recall"] = recall_score(all_labels, all_preds)
        metrics["f1"] = f1_score(all_labels, all_preds)
        
        cm = confusion_matrix(all_labels, all_preds)
        # tn (fake->fake), fp (fake->real), fn (real->fake), tp (real->real)
        tn, fp, fn, tp = cm.ravel()
        metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        # Fallback if batch only contains one class
        metrics["ap"] = 0.0
        metrics["roc_auc"] = 0.0
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1"] = 0.0
        metrics["confusion_matrix"] = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
        metrics["fpr"] = 0.0

    return metrics, all_labels, all_probs
