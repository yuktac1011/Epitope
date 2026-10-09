import argparse
import sys
import os
import uuid
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from app.core.config import ExperimentConfig
from app.core.metrics import calculate_threshold_from_validation, evaluate_metrics, aggregate_macro_metrics
from app.core.results import save_run_results, append_to_results_csv
from app.data.splits import load_fixed_splits
from app.models.autoencoder import FraudAutoencoder
from app.models.hybrid_model import HybridFraudModel
from app.federated.server import aggregate_weights
from app.federated.experiments import data_boundary_audit

def get_parameters(net):
    return [val.cpu().detach().numpy() for name, val in net.named_parameters() if val.requires_grad]

def set_parameters(net, parameters):
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    for param, new_val in zip(trainable_params, parameters):
        param.data = torch.tensor(new_val, dtype=param.dtype)

def train_local(net, X_train, y_train, epochs, seed):
    torch.manual_seed(seed)
    # Convert to tensors
    X_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    
    # Calculate pos_weight
    num_neg = (y_t == 0).sum()
    num_pos = (y_t == 1).sum()
    pos_weight = torch.tensor([num_neg / num_pos if num_pos > 0 else 1.0], dtype=torch.float32)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(net.get_trainable_parameters() if hasattr(net, 'get_trainable_parameters') else net.parameters(), lr=1e-3, weight_decay=1e-4)
    
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    net.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = net(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

def evaluate_model(net, X, y):
    net.eval()
    X_t = torch.tensor(X.values, dtype=torch.float32)
    with torch.no_grad():
        logits = net(X_t)
        probs = torch.sigmoid(logits).numpy().flatten()
    return y.values, probs

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def run_experiment(config: ExperimentConfig, drop_fraud_col: str = None):
    print(f"Starting run with config: {config.model_dump()}")
    git_commit = get_git_commit()
    run_id = str(uuid.uuid4())[:8]
    
    # Load autoencoder weights if hybrid
    ae_weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/best_autoencoder.pth'))
    
    bank_data = {}
    for bid in config.bank_ids:
        bank_data[bid] = load_fixed_splits(bid)
        
    input_dim = bank_data[config.bank_ids[0]]['train'][0].shape[1]
    latent_dim = 16
    
    results_by_seed = {}
    bytes_sent_by_seed = {}
    
    for seed in config.seeds:
        print(f"--- Running Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Instantiate networks for each bank
        nets = {}
        for bid in config.bank_ids:
            if config.model == "hybrid":
                ae = FraudAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
                if os.path.exists(ae_weights_path):
                    ae.load_state_dict(torch.load(ae_weights_path))
                net = HybridFraudModel(encoder=ae.encoder, input_dim=input_dim, latent_dim=latent_dim, concat_input=True)
            else:
                # Basic MLP
                net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
            nets[bid] = net

        bytes_sent = {bid: [] for bid in config.bank_ids}
            
        if config.method == "local":
            # Train purely locally
            for bid in config.bank_ids:
                X_tr, y_tr = bank_data[bid]['train']
                if drop_fraud_col:
                    mask = ~((y_tr == 1) & (X_tr[drop_fraud_col] > 0))
                    X_tr, y_tr = X_tr[mask], y_tr[mask]
                train_local(nets[bid], X_tr, y_tr, config.local_epochs * config.rounds, seed)
        elif config.method == "fedavg":
            # Initialize global model
            global_weights = get_parameters(nets[config.bank_ids[0]])
            for r in range(config.rounds):
                round_weights = []
                for bid in config.bank_ids:
                    set_parameters(nets[bid], global_weights)
                    X_tr, y_tr = bank_data[bid]['train']
                    if drop_fraud_col:
                        mask = ~((y_tr == 1) & (X_tr[drop_fraud_col] > 0))
                        X_tr, y_tr = X_tr[mask], y_tr[mask]
                    train_local(nets[bid], X_tr, y_tr, config.local_epochs, seed)
                    new_weights = get_parameters(nets[bid])
                    
                    # Audit
                    if r == 0:
                        data_boundary_audit(new_weights)
                        
                    # Calculate mock bytes sent (4 bytes per float parameter)
                    bytes_transferred = sum(w.size * 4 for w in new_weights)
                    bytes_sent[bid].append(bytes_transferred)
                    
                    num_examples = len(bank_data[bid]['train'][0])
                    round_weights.append((new_weights, num_examples))
                
                global_weights = aggregate_weights(round_weights)
            
            # Set all clients to global weights
            for bid in config.bank_ids:
                set_parameters(nets[bid], global_weights)
                
        bytes_sent_by_seed[seed] = bytes_sent
        
        # Evaluate
        seed_results = {}
        for bid in config.bank_ids:
            # 1. Evaluate on validation to pick threshold
            y_val, val_probs = evaluate_model(nets[bid], *bank_data[bid]['val'])
            threshold = calculate_threshold_from_validation(y_val, val_probs, target_fpr=0.01)
            
            # 2. Apply on test
            y_test, test_probs = evaluate_model(nets[bid], *bank_data[bid]['test'])
            metrics = evaluate_metrics(y_test, test_probs, threshold)
            seed_results[bid] = metrics
            
        # Also compute macro average
        seed_results["macro"] = aggregate_macro_metrics({k: v for k, v in seed_results.items() if k != "macro"})
        results_by_seed[seed] = seed_results
        
        # Append to CSV
        append_to_results_csv(run_id, config.model_dump(), {k: v for k, v in seed_results.items() if k != "macro"}, seed)

    save_run_results(run_id, config.model_dump(), git_commit, results_by_seed, bytes_sent_by_seed)
    print(f"Run {run_id} completed successfully!")
    return results_by_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    config = ExperimentConfig.from_yaml(args.config)
    run_experiment(config)
