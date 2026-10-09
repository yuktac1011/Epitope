import os
import sys
import torch
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from backend.app.data.loader import get_raw_dataset
from backend.app.federated.client import get_client_fn

def aggregate_weights(weights_list):
    """Simple Federated Averaging (FedAvg)."""
    # Create empty list of aggregated weights
    aggregated = []
    
    # Calculate the total number of examples across all clients
    total_examples = sum([num_examples for _, num_examples in weights_list])
    
    # Iterate through each layer
    num_layers = len(weights_list[0][0])
    for layer_idx in range(num_layers):
        layer_sum = None
        for client_weights, num_examples in weights_list:
            weighted_layer = client_weights[layer_idx] * (num_examples / total_examples)
            if layer_sum is None:
                layer_sum = weighted_layer
            else:
                layer_sum += weighted_layer
        aggregated.append(layer_sum)
        
    return aggregated

def run_manual_simulation(num_clients: int, num_rounds: int, df, ae_weights_path):
    print(f"\n{'='*60}")
    print(f"Starting Manual Federated Learning Simulation with {num_clients} Banks")
    print(f"Bypassing Flower's Ray requirement due to Python 3.13")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # 1. Initialize Clients
    client_factory = get_client_fn(df, ae_weights_path, as_numpy=True)
    clients = [client_factory(str(i)) for i in range(num_clients)]
    
    # 2. Get initial global weights from Client 0 (as a starting point)
    global_weights = clients[0].get_parameters({})
    
    # 3. Training Loop
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Federated Round {round_num}/{num_rounds} ---")
        
        round_weights = []
        round_metrics = []
        
        # A. Fit and Evaluate each Client
        for i, client in enumerate(clients):
            print(f"  Training Bank {i}...")
            # Train locally (inject global weights -> train -> return updated weights)
            updated_weights, num_examples, _ = client.fit(global_weights, {"local_epochs": 1})
            round_weights.append((updated_weights, num_examples))
            
            # Evaluate locally (just testing the new updated weights on local test set)
            loss, test_examples, metrics = client.evaluate(updated_weights, {})
            metrics["num_examples"] = test_examples
            round_metrics.append(metrics)
            
        # B. Aggregate Weights (FedAvg)
        global_weights = aggregate_weights(round_weights)
        
        # C. Aggregate Metrics
        total_test_examples = sum([m["num_examples"] for m in round_metrics])
        agg_ap = sum([m["ap"] * m["num_examples"] for m in round_metrics]) / total_test_examples
        agg_pr_auc = sum([m["pr_auc"] * m["num_examples"] for m in round_metrics]) / total_test_examples
        agg_rec = sum([m["recall_at_1_fpr"] * m["num_examples"] for m in round_metrics]) / total_test_examples
        
        print(f"  [Global Model] Avg Precision: {agg_ap:.4f} | PR-AUC: {agg_pr_auc:.4f} | Recall @ 1% FPR: {agg_rec:.4f}")

    end_time = time.time()
    
    print(f"\n--- Simulation completed in {end_time - start_time:.2f} seconds ---")
    
    estimated_mlp_params = 5000 
    bytes_per_param = 4
    comm_size_kb = (estimated_mlp_params * bytes_per_param * 2 * num_rounds * num_clients) / 1024
    print(f"Estimated Total Communication Payload: ~{comm_size_kb:.2f} KB")

if __name__ == "__main__":
    df = get_raw_dataset()
    if 'BankID' not in df.columns:
        np.random.seed(42)
        df['BankID'] = np.random.randint(0, 5, size=len(df))
        
    ae_weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/best_autoencoder.pth'))
    
    # Run Experiment 1: 2 Banks, 5 Rounds
    run_manual_simulation(num_clients=2, num_rounds=5, df=df, ae_weights_path=ae_weights_path)
    
    # Run Experiment 2: 3 Banks, 5 Rounds
    run_manual_simulation(num_clients=3, num_rounds=5, df=df, ae_weights_path=ae_weights_path)
