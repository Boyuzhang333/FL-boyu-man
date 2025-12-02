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
    Metrics,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------- Parameter parsing -----------------
parser = argparse.ArgumentParser(description="Flower Attack1 + Defense")
parser.add_argument("--round", type=int, default=10)
parser.add_argument("--data_split", type=str, default="iid")
parser.add_argument("--attack_type", type=str, default="label_flipping")
parser.add_argument("--defense", type=str, default="none", choices=["none","median","trimmed"])
parser.add_argument("--n_mal", type=int, default=0)
parser.add_argument("--run_id", type=int, default=0)
args = parser.parse_args()

rounds = args.round
metrics_history = []


# ----------------- fit_config -----------------
def fit_config(server_round: int):
    return {"server_round": server_round}


# ----------------- CNN Model -----------------
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
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(testloader, "Testing"):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            total_loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()

    acc = correct / len(testloader.dataset)
    return total_loss, acc


# ----------------- Evaluation per round -----------------
def evaluate_function(data_split):
    def evaluate(server_round, parameters, config):

        net = Net().to(DEVICE)
        params = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params})
        net.load_state_dict(state_dict, strict=True)

        _, _, testloader = prepare_dataset.load_datasets(2, "CIFAR10", data_split)
        loss, acc = test(net, testloader)

        print(f"[Server] Round {server_round}  loss={loss:.4f}  acc={acc:.4f}")

        metrics_history.append({"round": server_round, "loss": loss, "accuracy": acc})

        return loss, {"accuracy": acc}

    return evaluate


# ================================================================
#                     Defense 1: FedMedian
# ================================================================
class FedMedian(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):

        # Each element in results is (client_proxy, FitRes)
        all_ndarrays = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        num_layers = len(all_ndarrays[0])
        aggregated = []

        for layer in range(num_layers):
            layer_stack = np.array([client[layer] for client in all_ndarrays])
            aggregated.append(np.median(layer_stack, axis=0))

        return ndarrays_to_parameters(aggregated), {}

# ================================================================
#                     Defense 2: FedTrimmedAvg
# ================================================================
class FedTrimmedAvg(fl.server.strategy.FedAvg):

    def __init__(self, trim_ratio=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trim_ratio = trim_ratio

    def aggregate_fit(self, server_round, results, failures):

        all_ndarrays = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        n = len(all_ndarrays)
        k = int(n * self.trim_ratio)

        num_layers = len(all_ndarrays[0])
        aggregated = []

        for layer in range(num_layers):

            layer_stack = np.array([client[layer] for client in all_ndarrays])

            sorted_vals = np.sort(layer_stack, axis=0)
            trimmed = sorted_vals[k : n - k]

            aggregated.append(np.mean(trimmed, axis=0))

        return ndarrays_to_parameters(aggregated), {}

# ================================================================
#                Select defense strategy
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
#                Run server and write to CSV
# ================================================================
if __name__ == "__main__":

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy
    )

    os.makedirs("results_attack1", exist_ok=True)

    filename = f"results_attack1/{args.attack_type}_{args.defense}_{args.data_split}_mal{args.n_mal}_run{args.run_id}.csv"

    print("Saving:", filename)

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "accuracy", "loss"])
        writer.writeheader()
        for item in metrics_history:
            writer.writerow(item)
