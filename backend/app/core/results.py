import json
import csv
import os
from pydantic import BaseModel

class ResultSchema(BaseModel):
    run_id: str
    method: str
    model: str
    n_banks: int
    banks_included: str
    bank_id: str
    seed: int
    metric: str
    value: float

def save_run_results(run_id: str, config: dict, commit_hash: str, results: dict, bytes_sent: dict):
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../results/runs"))
    os.makedirs(out_dir, exist_ok=True)
    out_dict = {
        "run_id": run_id,
        "config_hash": str(hash(json.dumps(config, sort_keys=True))),
        "git_commit": commit_hash,
        "config": config,
        "results": results,
        "bytes_sent": bytes_sent
    }
    with open(os.path.join(out_dir, f"{run_id}.json"), "w") as f:
        json.dump(out_dict, f, indent=2)

def append_to_results_csv(run_id: str, config: dict, results_dict: dict, seed: int):
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../results/results.csv"))
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    file_exists = os.path.isfile(file_path)
    
    rows = []
    n_banks = len(config["bank_ids"])
    banks_included = "-".join(map(str, config["bank_ids"]))
    
    for bank_id, metrics in results_dict.items():
        for metric, value in metrics.items():
            row = ResultSchema(
                run_id=run_id,
                method=config["method"],
                model=config["model"],
                n_banks=n_banks,
                banks_included=banks_included,
                bank_id=str(bank_id),
                seed=seed,
                metric=metric,
                value=float(value)
            )
            rows.append(row.model_dump())
            
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ResultSchema.model_fields.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
