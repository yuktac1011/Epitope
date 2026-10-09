import os
import json
import itertools
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from typing import List

from app.core.config import ExperimentConfig
from app.run import run_experiment
from app.data.splits import load_fixed_splits

def paired_bootstrap(metric_a, metric_b, n_bootstrap=1000):
    diffs = np.array(metric_a) - np.array(metric_b)
    means = [np.mean(np.random.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_bootstrap)]
    return np.mean(diffs), np.percentile(means, 2.5), np.percentile(means, 97.5)

def get_coalition_results(coalition: tuple, seeds: List[int], cache_dir: str):
    """Run or load cached results for a specific coalition of banks."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"coalition_{'-'.join(map(str, coalition)) if coalition else 'empty'}.json")
    
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
            
    if not coalition:
        # Empty coalition baseline is usually a random model or 0
        res = {str(s): {"macro": {"ap": 0.0}} for s in seeds}
    else:
        config = ExperimentConfig(
            method="fedavg", model="hybrid", bank_ids=list(coalition), rounds=5, local_epochs=1, seeds=seeds
        )
        run_res = run_experiment(config)
        res = {str(s): run_res[s] for s in seeds}
        
    # Ensure all keys are strings to match JSON load behavior
    res = json.loads(json.dumps(res))
        
    with open(cache_file, "w") as f:
        json.dump(res, f)
    return res

def run_contributions():
    seeds = [42, 43, 44]
    banks = [0, 1, 2, 3, 4]
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/cache"))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/tables"))
    os.makedirs(out_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Per-bank deltas (Fed vs Local)
    # ---------------------------------------------------------
    print("Computing per-bank deltas...")
    # Get local baseline
    local_res = {}
    for bid in banks:
        config = ExperimentConfig(method="local", model="hybrid", bank_ids=[bid], rounds=5, local_epochs=1, seeds=seeds)
        local_res[bid] = run_experiment(config)
        
    fed_configs = {"Fed-2": [0,1], "Fed-3": [0,1,2], "Fed-5": banks}
    benefit_rows = []
    
    for fed_name, f_banks in fed_configs.items():
        fed_res = get_coalition_results(tuple(f_banks), seeds, cache_dir)
        
        for bid in banks:
            if bid not in f_banks:
                continue
            
            # Compare Fed AP vs Local AP
            fed_aps = [fed_res[str(s)][str(bid)]["ap"] for s in seeds]
            loc_aps = [local_res[bid][s][bid]["ap"] for s in seeds]
            
            mean_diff, lower, upper = paired_bootstrap(fed_aps, loc_aps)
            label = "Gains" if lower > 0 else ("Loses" if upper < 0 else "Neutral")
            
            benefit_rows.append({
                "Bank": bid,
                "Federation": fed_name,
                "Mean Delta (AP)": mean_diff,
                "95% CI Lower": lower,
                "95% CI Upper": upper,
                "Outcome": label
            })
            
    pd.DataFrame(benefit_rows).to_csv(os.path.join(out_dir, "benefit.csv"), index=False)
    
    # ---------------------------------------------------------
    # 2 & 3. Leave-one-bank-out & Exact Shapley
    # ---------------------------------------------------------
    print("Computing Shapley values across all 31 coalitions...")
    # Precompute all 32 coalitions (including empty)
    all_coalitions = []
    for r in range(len(banks) + 1):
        for combo in itertools.combinations(banks, r):
            all_coalitions.append(combo)
            get_coalition_results(combo, seeds, cache_dir)
            
    shapley_values = {b: 0.0 for b in banks}
    import math
    
    # Evaluate marginal contributions
    n = len(banks)
    for combo in all_coalitions:
        if not combo: continue
        
        combo_res = get_coalition_results(combo, seeds, cache_dir)
        combo_macro = np.mean([combo_res[str(s)]["macro"]["ap"] for s in seeds])
        
        for bid in combo:
            subset_without = tuple(x for x in combo if x != bid)
            without_res = get_coalition_results(subset_without, seeds, cache_dir)
            without_macro = np.mean([without_res[str(s)]["macro"]["ap"] for s in seeds])
            
            marginal_contrib = combo_macro - without_macro
            k = len(combo)
            weight = (math.factorial(k - 1) * math.factorial(n - k)) / math.factorial(n)
            
            shapley_values[bid] += weight * marginal_contrib
            
    shapley_df = pd.DataFrame([{"Bank": k, "Shapley Value (Macro AP)": v} for k, v in shapley_values.items()])
    shapley_df.to_csv(os.path.join(out_dir, "contribution.csv"), index=False)
    
    # Acceptance Test: Shapley sum check
    full_res = get_coalition_results(tuple(banks), seeds, cache_dir)
    empty_res = get_coalition_results((), seeds, cache_dir)
    full_macro = np.mean([full_res[str(s)]["macro"]["ap"] for s in seeds])
    empty_macro = np.mean([empty_res[str(s)]["macro"]["ap"] for s in seeds])
    
    assert abs(sum(shapley_values.values()) - (full_macro - empty_macro)) < 1e-4, "Shapley values do not sum to total delta!"

    # ---------------------------------------------------------
    # 4. Bank Profiles (JS Divergence & Fraud Rate)
    # ---------------------------------------------------------
    print("Generating Bank Profiles...")
    profiles = []
    
    # For global pooled distribution, load all splits
    all_train_dfs = [load_fixed_splits(b)["train"][0] for b in banks]
    pooled_df = pd.concat(all_train_dfs)
    pooled_types = pooled_df[[c for c in pooled_df.columns if c.startswith('type_')]].mean(axis=0)
    
    for bid in banks:
        data = load_fixed_splits(bid)["train"]
        X, y = data
        fraud_rate = y.mean()
        
        # Categorical divergence (Jensen-Shannon on transaction types)
        bank_types = X[[c for c in X.columns if c.startswith('type_')]].mean(axis=0)
        js_div = jensenshannon(bank_types, pooled_types)
        
        # Numeric divergence (KS on amounts)
        ks_stat, _ = ks_2samp(X['amount'], pooled_df['amount'])
        
        profiles.append({
            "Bank": bid,
            "Rows": len(X),
            "Fraud Rate": fraud_rate,
            "JS Divergence (Types)": js_div,
            "KS Stat (Amount)": ks_stat
        })
        
    pd.DataFrame(profiles).to_csv(os.path.join(out_dir, "profiles.csv"), index=False)
    
    # ---------------------------------------------------------
    # 5. Claim check & 6. Findings Markdown
    # ---------------------------------------------------------
    def check_claim(statement: str):
        # Extremely basic NLP to parse "Bank X [benefits/loses] in Fed-Y"
        import re
        match = re.search(r"Bank (\d+) (benefits|loses)(?: in (Fed-\d+))?", statement.lower())
        if not match: return "Cannot Parse"
        
        bid = int(match.group(1))
        action = match.group(2)
        target_fed = match.group(3) if match.group(3) else "fed-5"
        
        for row in benefit_rows:
            if row["Bank"] == bid and row["Federation"].lower() == target_fed:
                if action == "benefits" and row["Outcome"] == "Gains": return "Supported"
                if action == "loses" and row["Outcome"] == "Loses": return "Supported"
                return "Not Supported"
        return "Not Supported"

    # Evaluate existing Bank 4 claim
    claim = "Bank 4 benefits in Fed-5"
    claim_result = check_claim(claim)
    
    with open(os.path.join(out_dir, "findings.md"), "w") as f:
        f.write("# Federation Gain/Loss Findings\n\n")
        f.write("## Benefit Analysis\n")
        
        gains = [r['Bank'] for r in benefit_rows if r['Outcome'] == 'Gains' and r['Federation'] == 'Fed-5']
        loses = [r['Bank'] for r in benefit_rows if r['Outcome'] == 'Loses' and r['Federation'] == 'Fed-5']
        
        f.write(f"In Fed-5, Banks {gains} gained statistically significant performance compared to local training. ")
        if loses:
            f.write(f"However, Banks {loses} experienced a degradation in performance (negative transfer).\n\n")
        else:
            f.write("No banks experienced statistically significant performance degradation.\n\n")
            
        f.write("## Shapley Contributions\n")
        best_bank = max(shapley_values, key=shapley_values.get)
        f.write(f"Bank {best_bank} provided the highest marginal contribution to the federation's overall macro precision.\n\n")
        
        f.write("## Claim Checks\n")
        f.write(f"- Claim: '{claim}' -> **{claim_result}**\n")
        
    print("\nSUCCESS! Contribution experiment finished.")

if __name__ == "__main__":
    run_contributions()
