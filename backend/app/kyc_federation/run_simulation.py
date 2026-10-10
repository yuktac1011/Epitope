import os
import sys
import time
import json
import torch
from collections import OrderedDict
import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg

# Ensure FedShield root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.app.kyc.model import get_kyc_model, freeze_backbone
from backend.app.kyc.data import get_dataloaders
from backend.app.kyc.federation_client import get_partition_dataloaders, KYCFlowerClient

def weighted_average(metrics):
    ap_list = [num_examples * m["ap"] for num_examples, m in metrics if "ap" in m]
    examples = [num_examples for num_examples, m in metrics if "ap" in m]
    if sum(examples) == 0:
        return {"ap": 0.0}
    return {"ap": round(sum(ap_list) / sum(examples), 4)}

def run_federated_simulation(num_rounds: int = 2, num_clients: int = 3):
    print(f"============================================================")
    print(f"Starting Flower Federated Learning Simulation")
    print(f"Institutions: {num_clients} Simulated Banks | Communication Rounds: {num_rounds}")
    print(f"============================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def client_fn(cid: str) -> fl.client.Client:
        node_id = int(cid)
        train_loader, val_loader = get_partition_dataloaders(node_id, num_clients)
        model = get_kyc_model(pretrained=True)
        model = freeze_backbone(model)
        return KYCFlowerClient(model, train_loader, val_loader, device).to_client()

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda server_round: {"local_epochs": 1, "lr": 1e-3}
    )

    t0 = time.time()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0}
    )
    duration = time.time() - t0

    print("\n================ FEDERATED LEARNING RESULTS ================")
    print(f"Total Simulation Duration: {duration:.2f} seconds")
    print("Round History (Evaluated Metrics):", history.metrics_distributed)

    # Save summary report
    report_dir = "backend/app/kyc/reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "federated_simulation_report.json")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "duration_seconds": round(duration, 2),
        "metrics_distributed": {
            k: [(int(r), float(v)) for r, v in vals]
            for k, vals in history.metrics_distributed.items()
        }
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Federated benchmark report saved to: {report_path}")

    return history

if __name__ == "__main__":
    run_federated_simulation(num_rounds=2, num_clients=3)
