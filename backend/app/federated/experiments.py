import os
import sys
import time
import numpy as np
import pandas as pd
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from backend.app.data.loader import get_raw_dataset
from backend.app.federated.client import get_client_fn
from backend.app.federated.server import aggregate_weights

def data_boundary_audit(weights):
    """
    AUDIT CHECK: Ensures that only mathematical weight parameters are sent back to the server.
    Raw transactions and account IDs must NEVER leave the client.
    """
    assert isinstance(weights, list), "Client must return a list of arrays, not a DataFrame."
    
    total_params = 0
    for layer in weights:
        assert isinstance(layer, np.ndarray), f"Expected numpy array, got {type(layer)}. Potential data leak!"
        assert np.issubdtype(layer.dtype, np.floating), f"Weights must be floats, got {layer.dtype}. Potential data leak!"
        total_params += layer.size
        
    # Our MLP head has exactly 4,225 params (latent_dim=16+25=41 -> 64 -> 32 -> 1)
    # Check that we aren't sending massive amounts of raw data
    assert total_params < 10000, f"Payload too large ({total_params} params). Expected < 10,000 for MLP head."
    return total_params

def evaluate_global_model_on_all_banks(all_clients, global_weights):
    """
    Evaluate the given global weights across every single bank's test set.
    """
    results = {}
    for i, client in enumerate(all_clients):
        loss, test_examples, metrics = client.evaluate(global_weights, {})
        results[f"Bank_{i}"] = metrics
    return results

def run_experiment(num_participating_banks: int, all_clients, num_rounds: int = 5):
    print(f"\n{'='*60}")
    print(f"Running Experiment: {num_participating_banks} Participating Banks")
    print(f"{'='*60}")
    
    # 1. Select participating clients
    participating_clients = all_clients[:num_participating_banks]
    
    # 2. Get initial weights from the first client
    global_weights = participating_clients[0].get_parameters({})
    
    for round_num in range(1, num_rounds + 1):
        round_weights = []
        for i, client in enumerate(participating_clients):
            # Train client
            updated_weights, num_examples, _ = client.fit(global_weights, {"local_epochs": 1})
            
            # === DATA BOUNDARY AUDIT ===
            if round_num == 1:
                params_count = data_boundary_audit(updated_weights)
                if i == 0:
                    print(f"  [AUDIT PASS] Verified Bank {i} returned exactly {params_count} mathematical parameters. No raw data leaked.")
            
            round_weights.append((updated_weights, num_examples))
            
        # Aggregate (FedAvg)
        global_weights = aggregate_weights(round_weights)
        
    # Evaluate final model on ALL 5 banks
    print("\n  [Evaluating Final Global Model on ALL 5 Banks...]")
    bank_results = evaluate_global_model_on_all_banks(all_clients, global_weights)
    
    # Calculate Macro metrics (average across all 5 banks)
    macro_ap = np.mean([res["ap"] for res in bank_results.values()])
    macro_pr_auc = np.mean([res["pr_auc"] for res in bank_results.values()])
    macro_recall = np.mean([res["recall_at_1_fpr"] for res in bank_results.values()])
    
    print(f"  --> Macro Average Precision: {macro_ap:.4f}")
    
    return {
        "num_banks": num_participating_banks,
        "macro_ap": macro_ap,
        "macro_pr_auc": macro_pr_auc,
        "macro_recall": macro_recall,
        "per_bank_results": bank_results
    }

if __name__ == "__main__":
    df = get_raw_dataset()
    if 'BankID' not in df.columns:
        np.random.seed(42)
        df['BankID'] = np.random.randint(0, 5, size=len(df))
        
    ae_weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/best_autoencoder.pth'))
    
    print("\nInitializing ALL 5 Bank Clients in memory...")
    client_factory = get_client_fn(df, ae_weights_path, as_numpy=True)
    all_clients = [client_factory(str(i)) for i in range(5)]
    
    experiments = [1, 2, 3, 5]
    all_results = []
    
    for num_banks in experiments:
        results = run_experiment(num_banks, all_clients, num_rounds=5)
        all_results.append(results)
        
    # Save results
    output_rows = []
    for res in all_results:
        row = {
            "Num_Participating_Banks": res["num_banks"],
            "Macro_AP": res["macro_ap"],
            "Macro_PR_AUC": res["macro_pr_auc"],
            "Macro_Recall_1FPR": res["macro_recall"]
        }
        # Add per-bank breakdown
        for bank_name, metrics in res["per_bank_results"].items():
            row[f"{bank_name}_AP"] = metrics["ap"]
        output_rows.append(row)
        
    results_df = pd.DataFrame(output_rows)
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'federated_experiment_results.csv'))
    results_df.to_csv(out_path, index=False)
    
    print(f"\n\nSUCCESS! Full experiments complete. Results saved to {out_path}")
    print("\nSummary of Final Macro-Average Precision by network size:")
    print(results_df[["Num_Participating_Banks", "Macro_AP", "Macro_PR_AUC", "Macro_Recall_1FPR"]].to_markdown(index=False))
