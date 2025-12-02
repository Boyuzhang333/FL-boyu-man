import flwr as fl
from flwr.common import parameters_to_ndarrays
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
from tqdm import tqdm
import prepare_dataset
import numpy as np
import csv
import os
import io

# ----------------- 设备 -----------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------- 参数解析 -----------------
parser = argparse.ArgumentParser(description="Flower Attack2 (Model Poisoning) + Defense")
parser.add_argument("--round", type=int, default=10)
parser.add_argument("--data_split", type=str, default="iid")        # iid / non_iid_class
parser.add_argument("--attack_type", type=str, default="model_poisoning")
parser.add_argument("--defense", type=str, default="none", choices=["none", "median", "trimmed"])
parser.add_argument("--n_mal", type=int, default=0)
parser.add_argument("--run_id", type=int, default=0)

args = parser.parse_args()
rounds = args.round
metrics_history = []

# ----------------- fit_config -----------------
def fit_config(server_round: int):
    return {"server_round": server_round}

# ----------------- 模型定义 -----------------
class Net(nn.Module):
    def __init__(self):
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

# ----------------- 测试函数 -----------------
def test(net, testloader):
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

# ----------------- 每轮评估 -----------------
def evaluate_function(data_split: str):
    def evaluate(server_round, parameters, config):
        net = Net().to(DEVICE)

        # Flower parameters → numpy arrays → state_dict
        params = parameters_to_ndarrays(parameters)
        params_dict = zip(net.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

        _, _, testloader = prepare_dataset.load_datasets(2, "CIFAR10", data_split)
        loss, accuracy = test(net, testloader)

        print(f"Round {server_round}: loss={loss} accuracy={accuracy}")

        metrics_history.append({"round": server_round, "loss": loss, "accuracy": accuracy})
        return loss, {"accuracy": accuracy}

    return evaluate

# ----------------- 防御策略：FedMedian -----------------
class FedMedian(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):

        all_weights = []
        for _, fit_res, _ in results:
            nds = parameters_to_ndarrays(fit_res.parameters)
            all_weights.append(nds)

        all_weights = np.array(all_weights, dtype=object)

        aggregated = []
        for i in range(len(all_weights[0])):
            layer_vals = np.array([client[i] for client in all_weights], dtype=float)
            aggregated.append(np.median(layer_vals, axis=0))

        return aggregated, {}

# ----------------- 防御策略：FedTrimmedAvg -----------------
class FedTrimmedAvg(fl.server.strategy.FedAvg):
    def __init__(self, trim_ratio=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trim_ratio = trim_ratio

    def aggregate_fit(self, server_round, results, failures):

        all_weights = []
        for _, fit_res, _ in results:
            nds = parameters_to_ndarrays(fit_res.parameters)
            all_weights.append(nds)

        all_weights = np.array(all_weights, dtype=object)
        k = int(len(all_weights) * self.trim_ratio)

        aggregated = []
        for i in range(len(all_weights[0])):
            layer_vals = np.array([client[i] for client in all_weights], dtype=float)
            sorted_vals = np.sort(layer_vals, axis=0)
            trimmed = sorted_vals[k : len(sorted_vals) - k]
            aggregated.append(np.mean(trimmed, axis=0))

        return aggregated, {}

# ----------------- 选择防御策略 -----------------
if args.defense == "none":
    strategy = fl.server.strategy.FedAvg(
        on_evaluate_config_fn=fit_config,
        on_fit_config_fn=fit_config,
        evaluate_fn=evaluate_function(args.data_split),
    )
elif args.defense == "median":
    strategy = FedMedian(
        on_evaluate_config_fn=fit_config,
        on_fit_config_fn=fit_config,
        evaluate_fn=evaluate_function(args.data_split),
    )
elif args.defense == "trimmed":
    strategy = FedTrimmedAvg(
        trim_ratio=0.2,
        on_evaluate_config_fn=fit_config,
        on_fit_config_fn=fit_config,
        evaluate_fn=evaluate_function(args.data_split),
    )

# ----------------- 启动服务器 + 保存 CSV -----------------
if __name__ == "__main__":

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

    # 保存结果
    os.makedirs("results_attack2", exist_ok=True)

    filename = f"results_attack2/{args.attack_type}_{args.defense}_{args.data_split}_mal{args.n_mal}_run{args.run_id}.csv"
    print(f"[Saving Attack2 Results] {filename}")

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "accuracy", "loss"])
        writer.writeheader()
        for item in metrics_history:
            writer.writerow(item)
