import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, 
    precision_recall_curve, 
    auc, 
    roc_curve, 
    precision_score, 
    recall_score, 
    f1_score
)
import sys
import os

# Add the root directory to the python path so imports work cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.app.data.loader import get_raw_dataset
from backend.app.features.preprocessing import preprocess_bank_data

def evaluate_model(y_true, y_pred, y_prob):
    # Average Precision
    ap = average_precision_score(y_true, y_prob)
    
    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Recall at 1% FPR
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Find the index where fpr is closest to 0.01 but not exceeding it
    valid_fpr_indices = np.where(fpr <= 0.01)[0]
    if len(valid_fpr_indices) > 0:
        idx = valid_fpr_indices[-1]
        recall_at_1_fpr = tpr[idx]
    else:
        recall_at_1_fpr = 0.0
    
    # Basic metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "Average Precision": ap,
        "PR-AUC": pr_auc,
        "Recall @ 1% FPR": recall_at_1_fpr,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

def run_local_baseline():
    df = get_raw_dataset()
    
    # If BankID doesn't exist in the raw dataset, we simulate it for 5 banks (0 to 4)
    if 'BankID' not in df.columns:
        print("\n'BankID' column not found. Assigning random BankIDs 0-4 for federated simulation.")
        np.random.seed(42)
        df['BankID'] = np.random.randint(0, 5, size=len(df))
    
    results = []
    
    for bank_id in sorted(df['BankID'].unique()):
        print(f"\n--- Training Local Baseline for Bank {bank_id} ---")
        bank_df = df[df['BankID'] == bank_id].copy()
        
        # Preprocess using our modular pipeline
        X_train, y_train, X_test, y_test = preprocess_bank_data(bank_df)
        
        print(f"Bank {bank_id} Training size: {len(X_train)} | Test size: {len(X_test)}")
        
        # Train Logistic Regression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict on Test Data
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Evaluate Metrics
        metrics = evaluate_model(y_test, y_pred, y_prob)
        metrics["BankID"] = bank_id
        results.append(metrics)
        
        for k, v in metrics.items():
            if k != "BankID":
                print(f"{k}: {v:.4f}")
                
    results_df = pd.DataFrame(results)
    
    # Reorder columns so BankID is first
    cols = ['BankID'] + [c for c in results_df.columns if c != 'BankID']
    results_df = results_df[cols]
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), "local_baseline_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved local baseline results to {output_path}")
    
    return results_df

if __name__ == "__main__":
    run_local_baseline()
