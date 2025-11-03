import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics
import argparse

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--round",
    type=int,
    default=10,
    help="Partition of the dataset divided into 3 iid partitions created artificially.",
)
rounds = parser.parse_args().round
# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    round = metrics[0][1]["round"]
    examples = [num_examples for num_examples, _ in metrics]
    accuracy = sum(accuracies) / sum(examples)
    print(f"Round {round} Global model test accuracy: {accuracy}")
    # Aggregate and return custom metric (weighted average)
    try:
        with open('log.txt', 'a') as f:
            if round == 1:
                f.write("\n-------------------------------------\n")
            f.write(str(accuracy)+" ")
    except FileNotFoundError:
        with open('log.txt', 'w') as f:
            if round == 1:
                f.write("\n-------------------------------------\n")
            f.write(str(accuracy)+" ")

    return {"accuracy": {accuracy}}


def fit_config(server_round:int):
    config = {
        "server_round": server_round,
    }
    return config


strategy = fl.server.strategy.FedAvg(
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average
)
fl.server.start_server(
    server_address="0.0.0.0:8123",
    config=fl.server.ServerConfig(num_rounds=rounds),
    strategy=strategy
)