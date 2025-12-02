import flwr as fl
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import csv
import os
import prepare_dataset

from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------- Parameter parsing -----------------
parser = argparse.ArgumentParser(description="Flower Attack2 + Defense")
parser.add_argument("--round", type=int, default=10)
parser.add_argument("--data_split", type=str, default="iid")
parser.add_argument("--attack_type", type=str, default="model_poisoning")
parser.add_argument("--defense", type=str, default="none", choices=["none","median","trimmed"])
parser.add_argument("--n_mal", type=int, default=0)
parser.add_argument("--run_id", type=int, default=0)
args = parser.parse_args()

rounds = args.round
metrics_history = []


# ----------------- fit_config -----------------
def fit_config(server_round: int):
    return {"server_round": server_round}


# ----------------- Model -----------------
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ----------------- Test function -----------------
def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    loss_total = 0

    with torch.no_grad():
        for x, y in tqdm(testloader, "Testing"):
            outputs = net(x.to(DEVICE))
            y = y.to(DEVICE)
            loss_total += criterion(outputs, y).item()
            correct += (outputs.argmax(1) == y).sum().item()

    acc = correct / len(testloader.dataset)
    return loss_total, acc


# ----------------- Server-side evaluation -----------------
def evaluate_function(data_split):
    def evaluate(server_round, parameters, config):

        # parameters is already a list of ndarrays, no need for parameters_to_ndarrays
        net = Net().to(DEVICE)
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

        _, _, testloader = prepare_dataset.load_datasets(2, "CIFAR10", data_split)
        loss, acc = test(net, testloader)

        print(f"[Server] Round {server_round}: loss={loss:.4f} acc={acc:.4f}")

        metrics_history.append({"round": server_round, "loss": loss, "accuracy": acc})
        return loss, {"accuracy": acc}

    return evaluate


# ================================================================
#                     Defense 1: FedMedian
# ================================================================
class FedMedian(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):

        # Attack2 results structure: [(client_proxy, FitRes)]
        all_layers = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        num_layers = len(all_layers[0])
        aggregated = []

        for layer in range(num_layers):
            stack = np.array([client[layer] for client in all_layers])
            aggregated.append(np.median(stack, axis=0))

        return ndarrays_to_parameters(aggregated), {}


# ================================================================
#                     Defense 2: FedTrimmedAvg
# ================================================================
class FedTrimmedAvg(fl.server.strategy.FedAvg):

    def __init__(self, trim_ratio=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trim_ratio = trim_ratio

    def aggregate_fit(self, server_round, results, failures):

        all_layers = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        n = len(all_layers)
        k = int(n * self.trim_ratio)

        num_layers = len(all_layers[0])
        aggregated = []

        for layer in range(num_layers):
            values = np.array([client[layer] for client in all_layers])
            sorted_vals = np.sort(values, axis=0)
            trimmed = sorted_vals[k:n-k]
            aggregated.append(trimmed.mean(axis=0))

        return ndarrays_to_parameters(aggregated), {}


# ================================================================
#                     Strategy selection
# ================================================================
if args.defense == "none":
    strategy = fl.server.strategy.FedAvg(
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
        evaluate_fn=evaluate_function(args.data_split),
    )

elif args.defense == "median":
    strategy = FedMedian(
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
        evaluate_fn=evaluate_function(args.data_split),
    )

elif args.defense == "trimmed":
    strategy = FedTrimmedAvg(
        trim_ratio=0.2,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
        evaluate_fn=evaluate_function(args.data_split),
    )


# ================================================================
#                     Start server
# ================================================================
if __name__ == "__main__":

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

    os.makedirs("results_attack2", exist_ok=True)
    filename = f"results_attack2/{args.attack_type}_{args.defense}_{args.data_split}_mal{args.n_mal}_run{args.run_id}.csv"

    print("Saving:", filename)

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "accuracy", "loss"])
        writer.writeheader()
        for item in metrics_history:
            writer.writerow(item)
