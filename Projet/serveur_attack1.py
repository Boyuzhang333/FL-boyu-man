import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
from tqdm import tqdm
import prepare_dataset

# For saving CSV results
import csv
import os

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameter parsing for Attack1 experiments
parser = argparse.ArgumentParser(description="Flower server Attack1 (label flipping)")
parser.add_argument(
    "--round",
    type=int,
    default=10,
    help="Number of FL rounds",
)
parser.add_argument(
    "--data_split",
    type=str,
    default="iid",
    help="Type of data split: iid or non_iid_class",
)
parser.add_argument(
    "--attack_type",
    type=str,
    default="label_flipping",
    help="Type of attack (for naming results)",
)
parser.add_argument(
    "--n_mal",
    type=int,
    default=0,
    help="Number of malicious clients (for naming results)",
)
parser.add_argument(
    "--run_id",
    type=int,
    default=0,
    help="Run id / repeat index (for naming results)",
)

args = parser.parse_args()
rounds = args.round

# Store metrics for each round (round, loss, accuracy)
metrics_history = []  # Append one record per round in evaluate_function


class Net(nn.Module):
    """Model"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
print(f"🎯 Attack1 Server Starting")
print(f"   Attack Type: {current_experiment_info['attack_type']}")
print(f"   Data Split: {current_experiment_info['data_split']}")
print(f"   Malicious Clients: {current_experiment_info['n_mal']}")
print(f"   Run ID: {current_experiment_info['run_id']}")
print(f"   Training Rounds: {current_experiment_info['rounds']}")
def test(net, testloader):
    """Validate the model on the validation set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader, "Testing"):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


# The `evaluate` function will be called by Flower after every round
# Modified as closure to pass data_split and record metrics_history
def evaluate_function(data_split: str):
    def evaluate(server_round, parameters, config):
        net = Net().to(DEVICE)
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

        # Use data_split (iid / non_iid_class)
        _, _, testloader = prepare_dataset.load_datasets(2, "CIFAR10", data_split)
        loss, accuracy = test(net, testloader)
        print(f"Round {server_round}: Server-side evaluation loss {loss} / accuracy {accuracy}")

        # Record one log entry here, to be written to CSV later
        metrics_history.append(
            {
                "round": server_round,
                "loss": loss,
                "accuracy": accuracy,
            }
        )

        return loss, {"accuracy": accuracy}

    return evaluate


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
        with open("log.txt", "a") as f:
            if round == 1:
                f.write("\n-------------------------------------\n")
            f.write(str(accuracy) + " ")
    except FileNotFoundError:
        with open("log.txt", "w") as f:
            if round == 1:
                f.write("\n-------------------------------------\n")
            f.write(str(accuracy) + " ")

    return {"accuracy": {accuracy}}


def fit_config(server_round: int):
    config = {
        "server_round": server_round,
    }
    return config


## Server defense (see Flower doc: https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.html)
### Your work below ###

strategy = fl.server.strategy.FedAvg(
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=fit_config,
    evaluate_fn=evaluate_function(args.data_split),  # Pass data_split
)

### Your work above ###


# Wrap server startup and CSV saving in main function, write CSV after training
if __name__ == "__main__":
    # Start FL server (training logic remains unchanged)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

    # After training, write metrics_history to CSV
    # Directory: results1 (create results1 folder)
    os.makedirs("results1", exist_ok=True)

    # Filename includes: attack_type, data_split, n_mal, run_id
    filename = f"results1/{args.attack_type}_{args.data_split}_mal{args.n_mal}_run{args.run_id}.csv"

    print(f"[Man / Attack1] Saving metrics to: {filename}")

    # Write CSV: columns are round, accuracy, loss (in your requested order)
    with open(filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "accuracy", "loss"])
        writer.writeheader()
        for item in metrics_history:
            writer.writerow(
                {
                    "round": item["round"],
                    "accuracy": item["accuracy"],
                    "loss": item["loss"],
                }
            )