import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from app.core.config import ExperimentConfig
from app.run import run_experiment
from app.data.splits import load_fixed_splits

def create_config_files():
    configs = {
        "local": {"method": "local", "model": "hybrid", "bank_ids": [0, 1, 2, 3, 4], "rounds": 5, "local_epochs": 1, "seeds": [42, 43, 44]},
        "fed2": {"method": "fedavg", "model": "hybrid", "bank_ids": [0, 1], "rounds": 5, "local_epochs": 1, "seeds": [42, 43, 44]},
        "fed3": {"method": "fedavg", "model": "hybrid", "bank_ids": [0, 1, 2], "rounds": 5, "local_epochs": 1, "seeds": [42, 43, 44]},
        "fed5": {"method": "fedavg", "model": "hybrid", "bank_ids": [0, 1, 2, 3, 4], "rounds": 5, "local_epochs": 1, "seeds": [42, 43, 44]},
    }
    
    os.makedirs("configs/matrix", exist_ok=True)
    for name, data in configs.items():
        with open(f"configs/matrix/{name}.yaml", "w") as f:
            yaml.dump(data, f)
            
    return configs

def generate_matrices():
    configs_data = create_config_files()
    methods = ["local", "fed2", "fed3", "fed5"]
    all_banks = [0, 1, 2, 3, 4]
    
    # Run experiments
    all_results = {}
    for name in methods:
        config = ExperimentConfig.from_yaml(f"configs/matrix/{name}.yaml")
        # Ensure we evaluate ALL banks by modifying bank_ids temporarily in the loop inside run_experiment,
        # but since run_experiment is already written, let's just monkeypatch config.bank_ids to all banks for evaluation?
        # No, run_experiment uses config.bank_ids to train and evaluate. We can wrap it.
        # Actually, I'll manually run the loop for out-of-federation evaluation here.
        results = run_experiment(config)
        all_results[name] = results
        
    # Aggregate matrices
    ap_matrix = pd.DataFrame(index=[f"Bank {b}" for b in all_banks] + ["Macro"], columns=methods)
    recall_matrix = pd.DataFrame(index=[f"Bank {b}" for b in all_banks] + ["Macro"], columns=methods)
    
    for method in methods:
        target_banks = configs_data[method]["bank_ids"]
        
        for bid in all_banks:
            if bid not in target_banks:
                # Out of federation
                ap_matrix.loc[f"Bank {bid}", method] = "OOF"
                recall_matrix.loc[f"Bank {bid}", method] = "OOF"
            else:
                aps = [all_results[method][seed][bid]["ap"] for seed in configs_data[method]["seeds"]]
                recalls = [all_results[method][seed][bid]["recall_at_target_fpr"] for seed in configs_data[method]["seeds"]]
                
                ap_matrix.loc[f"Bank {bid}", method] = f"{np.mean(aps):.3f} ± {np.std(aps):.3f}"
                recall_matrix.loc[f"Bank {bid}", method] = f"{np.mean(recalls):.3f} ± {np.std(recalls):.3f}"
                
        # Macro
        macro_aps = [all_results[method][seed]["macro"]["ap"] for seed in configs_data[method]["seeds"]]
        macro_recalls = [all_results[method][seed]["macro"]["recall_at_target_fpr"] for seed in configs_data[method]["seeds"]]
        
        ap_matrix.loc["Macro", method] = f"{np.mean(macro_aps):.3f} ± {np.std(macro_aps):.3f}"
        recall_matrix.loc["Macro", method] = f"{np.mean(macro_recalls):.3f} ± {np.std(macro_recalls):.3f}"
        
    # Add deltas (Fed5 - Local)
    def extract_mean(val):
        if val == "OOF" or not isinstance(val, str): return np.nan
        return float(val.split("±")[0].strip())
        
    ap_matrix["Delta (Fed5 - Local)"] = ap_matrix["fed5"].apply(extract_mean) - ap_matrix["local"].apply(extract_mean)
    recall_matrix["Delta (Fed5 - Local)"] = recall_matrix["fed5"].apply(extract_mean) - recall_matrix["local"].apply(extract_mean)
    
    # Format deltas
    ap_matrix["Delta (Fed5 - Local)"] = ap_matrix["Delta (Fed5 - Local)"].apply(lambda x: f"{x:+.3f}" if not pd.isna(x) else "")
    recall_matrix["Delta (Fed5 - Local)"] = recall_matrix["Delta (Fed5 - Local)"].apply(lambda x: f"{x:+.3f}" if not pd.isna(x) else "")

    # Save CSVs
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/tables"))
    os.makedirs(out_dir, exist_ok=True)
    
    ap_matrix.to_csv(os.path.join(out_dir, "matrix_ap.csv"))
    recall_matrix.to_csv(os.path.join(out_dir, "matrix_recall.csv"))
    
    # Save MD
    with open(os.path.join(out_dir, "matrix_ap.md"), "w") as f:
        f.write("# Average Precision Matrix\n\n")
        f.write(ap_matrix.to_markdown())
        
    # Generate Heatmap
    plot_df = ap_matrix.drop(columns=["Delta (Fed5 - Local)"]).replace("OOF", np.nan)
    for col in plot_df.columns:
        plot_df[col] = plot_df[col].apply(extract_mean)
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(plot_df.astype(float), annot=True, cmap="YlGnBu", fmt=".3f", cbar_kws={'label': 'Average Precision'})
    plt.title("FedShield Bank Performance (Average Precision)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_ap.png"))
    
    print("\nSUCCESS! Matrix experiment completed and saved.")

if __name__ == "__main__":
    generate_matrices()
