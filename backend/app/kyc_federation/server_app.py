from typing import List, Tuple
import flwr as fl
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation results from clients."""
    # Multiply accuracy/ap of each client by number of examples used
    ap_list = [num_examples * m["ap"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"ap": sum(ap_list) / sum(examples)}

def server_fn(context: fl.common.Context) -> fl.server.ServerAppComponents:
    num_rounds = context.run_config.get("num-server-rounds", context.run_config.get("num_rounds", 2))
    fraction_fit = context.run_config.get("fraction-fit", context.run_config.get("fraction_fit", 1.0))
    
    # Configure FedAvg strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda server_round: {"local_epochs": 1, "lr": 1e-3}
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    
    return fl.server.ServerAppComponents(strategy=strategy, config=config)

# Create the ServerApp
app = ServerApp(server_fn=server_fn)
