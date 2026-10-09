import os
import sys
import numpy as np
import pandas as pd
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from backend.app.data.loader import get_raw_dataset
from backend.app.federated.client import get_client_fn

def run_local_hybrid_baselines():
    print("Loading data...")
    df = get_raw_dataset()
    if 'BankID' not in df.columns:
        np.random.seed(42)
        df['BankID'] = np.random.randint(0, 5, size=len(df))
        
    ae_weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'best_autoencoder.pth'))
    
    # We use our federated client factory because it perfectly isolates the datasets per bank 
    # and handles all preprocessing identically.
    client_factory = get_client_fn(df, ae_weights_path, as_numpy=True)
    
    results = []
    
    for bank_id in range(5):
        print(f"\n{'='*50}")
        print(f"Training Local Hybrid Model for Bank {bank_id}")
        print(f"{'='*50}")
        
        # Initialize client for specific bank
        client = client_factory(str(bank_id))
        
        # Start from the newly initialized weights
        current_weights = client.get_parameters({})
        
        # Train completely locally for 5 epochs (matches the 5 rounds used in the FL simulation)
        print("Training locally for 5 epochs...")
        updated_weights, _, _ = client.fit(current_weights, {"local_epochs": 5})
        
        # Evaluate locally
        print("Evaluating on local test set...")
        loss, num_examples, metrics = client.evaluate(updated_weights, {})
        
        print(f"Bank {bank_id} Results -> AP: {metrics['ap']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Recall @ 1% FPR: {metrics['recall_at_1_fpr']:.4f}")
        
        results.append({
            "BankID": bank_id,
            "Average Precision": metrics["ap"],
            "PR-AUC": metrics["pr_auc"],
            "Recall @ 1% FPR": metrics["recall_at_1_fpr"]
        })
        
    df_res = pd.DataFrame(results)
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'local_hybrid_per_bank.csv'))
    df_res.to_csv(out_path, index=False)
    
    print(f"\n\nSUCCESS! All isolated local models trained.")
    print(f"Results saved to: {out_path}\n")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    run_local_hybrid_baselines()
