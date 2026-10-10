import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
except ImportError:
    pass

from app.core.config import ExperimentConfig
from app.run import run_experiment, get_parameters, set_parameters, evaluate_model
from app.core.metrics import calculate_threshold_from_validation, evaluate_metrics, aggregate_macro_metrics
from app.data.splits import load_fixed_splits
from app.models.autoencoder import FraudAutoencoder
from app.models.hybrid_model import HybridFraudModel
from app.federated.server import aggregate_weights

def run_dp_experiment(config: ExperimentConfig, target_epsilon: float):
    # Setup similar to run.py but stateful PrivacyEngine per client
    seeds = config.seeds
    bank_ids = config.bank_ids
    
    bank_data = {}
    for bid in bank_ids:
        bank_data[bid] = load_fixed_splits(bid)
        
    input_dim = bank_data[bank_ids[0]]['train'][0].shape[1]
    latent_dim = 16
    ae_weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/best_autoencoder.pth'))
    
    all_results = {}
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        nets = {}
        privacy_engines = {}
        optimizers = {}
        loaders = {}
        
        for bid in bank_ids:
            if config.model == "hybrid":
                ae = FraudAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
                if os.path.exists(ae_weights_path):
                    ae.load_state_dict(torch.load(ae_weights_path))
                net = HybridFraudModel(encoder=ae.encoder, input_dim=input_dim, latent_dim=latent_dim, concat_input=True)
            else:
                net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
                
            # Replace BN if needed (none are currently used, but standard Opacus practice)
            net = ModuleValidator.fix(net)
            
            X_tr, y_tr = bank_data[bid]['train']
            X_t = torch.tensor(X_tr.values, dtype=torch.float32)
            y_t = torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1)
            dataset = torch.utils.data.TensorDataset(X_t, y_t)
            loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
            
            # Setup Optimizer
            opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            
            pe = None
            if target_epsilon != float('inf'):
                pe = PrivacyEngine()
                net, opt, loader = pe.make_private_with_epsilon(
                    module=net,
                    optimizer=opt,
                    data_loader=loader,
                    target_epsilon=target_epsilon,
                    target_delta=1.0 / (10 * len(dataset)),
                    epochs=config.rounds * config.local_epochs,
                    max_grad_norm=config.dp.max_grad_norm
                )
                
            nets[bid] = net
            optimizers[bid] = opt
            loaders[bid] = loader
            privacy_engines[bid] = pe

        # Federated Loop
        global_weights = get_parameters(nets[bank_ids[0]])
        for r in range(config.rounds):
            round_weights = []
            for bid in bank_ids:
                set_parameters(nets[bid], global_weights)
                
                # Train
                net = nets[bid]
                opt = optimizers[bid]
                loader = loaders[bid]
                
                num_neg = (y_t == 0).sum()
                num_pos = (y_t == 1).sum()
                pos_weight = torch.tensor([num_neg / num_pos if num_pos > 0 else 1.0], dtype=torch.float32)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                
                net.train()
                for _ in range(config.local_epochs):
                    for batch_x, batch_y in loader:
                        opt.zero_grad()
                        logits = net(batch_x)
                        loss = criterion(logits, batch_y)
                        loss.backward()
                        opt.step()
                        
                # Ensure epsilon didn't decrease
                if privacy_engines[bid] is not None:
                    eps = privacy_engines[bid].get_epsilon(delta=1.0 / (10 * len(bank_data[bid]['train'][0])))
                    print(f"Bank {bid} Cumulative Epsilon after round {r+1}: {eps:.2f}")

                new_weights = get_parameters(nets[bid])
                round_weights.append((new_weights, len(loader.dataset)))
            
            global_weights = aggregate_weights(round_weights)
            
        for bid in bank_ids:
            set_parameters(nets[bid], global_weights)
            
        # Eval
        seed_results = {}
        for bid in bank_ids:
            # Need to evaluate the original un-wrapped model structure for simplicity, 
            # or just call net directly. Opacus wrapper (GradSampleModule) supports eval directly.
            y_val, val_probs = evaluate_model(nets[bid], *bank_data[bid]['val'])
            threshold = calculate_threshold_from_validation(y_val, val_probs, target_fpr=0.01)
            y_test, test_probs = evaluate_model(nets[bid], *bank_data[bid]['test'])
            seed_results[bid] = evaluate_metrics(y_test, test_probs, threshold)
            
        seed_results["macro"] = aggregate_macro_metrics({k: v for k, v in seed_results.items()})
        all_results[seed] = seed_results
        
    return all_results

def run_dp_sweep():
    epsilons = [1.0, 3.0, 8.0, float('inf')]
    seeds = [42, 43, 44]
    banks = [0, 1, 2, 3, 4]
    
    config = ExperimentConfig(
        method="fedavg", model="mlp", bank_ids=banks, rounds=5, local_epochs=1, seeds=seeds
    )
    
    results_by_eps = {}
    for eps in epsilons:
        print(f"\n--- Running Target Epsilon: {eps} ---")
        results_by_eps[eps] = run_dp_experiment(config, target_epsilon=eps)
        
    # Aggregate and Save
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/tables"))
    os.makedirs(out_dir, exist_ok=True)
    
    rows = []
    for eps, seeds_res in results_by_eps.items():
        for bid in banks + ["macro"]:
            aps = [seeds_res[s][bid]["ap"] for s in seeds]
            recalls = [seeds_res[s][bid]["recall_at_target_fpr"] for s in seeds]
            
            rows.append({
                "Epsilon": eps if eps != float('inf') else "No DP",
                "Bank/Scope": bid,
                "Mean AP": np.mean(aps),
                "Std AP": np.std(aps),
                "Mean Recall@1%FPR": np.mean(recalls),
                "Std Recall@1%FPR": np.std(recalls)
            })
            
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "dp.csv"), index=False)
    
    # Plot Epsilon vs Score (Macro AP)
    macro_df = df[df["Bank/Scope"] == "macro"].copy()
    macro_df["Epsilon_num"] = macro_df["Epsilon"].replace("No DP", 15.0) # plot No DP at end
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(macro_df["Epsilon_num"], macro_df["Mean AP"], yerr=macro_df["Std AP"], marker='o', capsize=5)
    plt.xticks(list(macro_df["Epsilon_num"]), list(macro_df["Epsilon"]))
    plt.xlabel('Cumulative Privacy Budget (Epsilon)')
    plt.ylabel('Macro Average Precision')
    plt.title('Privacy-Utility Trade-off in FedShield')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dp_tradeoff.png"))
    
    # Save Guarantee text
    with open(os.path.join(out_dir, "dp_guarantee.md"), "w") as f:
        f.write("# Differential Privacy Guarantee for FedShield\n\n")
        f.write("This prototype implements **Record-Level Differential Privacy** using DP-SGD (via Opacus) at each individual client.\n\n")
        f.write("## What is Protected\n")
        f.write("- **Record-level privacy**: The presence or absence of any single transaction (record) within a bank's local dataset is mathematically obscured.\n")
        f.write("- **Honest-but-curious server**: The central server only receives clipped and noised gradients. It cannot reverse-engineer individual transactions from the model updates.\n\n")
        f.write("## What is NOT Protected (Limitations)\n")
        f.write("- **No Secure Aggregation (SecAgg)**: Model weights are transmitted in the clear over the network to the server. An active man-in-the-middle could inspect the noised weights.\n")
        f.write("- **Client-level privacy**: The DP guarantee applies to individual rows within a bank, not the entire bank's participation. The server still knows which banks participated.\n")
        
    print("\nSUCCESS! DP sweep completed.")

if __name__ == "__main__":
    run_dp_sweep()
