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

# 👉 新增：用于保存 CSV
import csv
import os

# 设备
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------- 参数解析（增加 Attack1 实验相关参数） -----------------
parser = argparse.ArgumentParser(description="Flower serveur Attack1 (label inversion)")
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

# 👉 新增：用于保存每一轮的 (round, loss, accuracy)
metrics_history = []  # 每轮在 evaluate_function 里 append 一条记录


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
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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


# The `evaluate` function will be by Flower called after every round
# 👉 修改成闭包，把 data_split 带进来，并在里面记录 metrics_history
def evaluate_function(data_split: str):
    def evaluate(server_round, parameters, config):
        net = Net().to(DEVICE)
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

        # 使用 data_split（iid / non_iid_class）
        _, _, testloader = prepare_dataset.load_datasets(2, "CIFAR10", data_split)
        loss, accuracy = test(net, testloader)
        print(f"Round {server_round}: Server-side evaluation loss {loss} / accuracy {accuracy}")

        # 👉 在这里记录一条日志，稍后写入 CSV
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


## Défense du serveur (voir Flower doc : https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.html)
### Your work below ###

strategy = fl.server.strategy.FedAvg(
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=fit_config,
    evaluate_fn=evaluate_function(args.data_split),  # 👉 传入 data_split
)

### Your work above ###


# 👉 把启动和 CSV 保存包一层 main，训练结束后写 CSV
if __name__ == "__main__":
    # 启动 FL 服务器（训练逻辑保持不变）
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

    # 训练结束后，把 metrics_history 写入 CSV
    # 目录：results1（你说可以建一个 result1 的文件，这里建一个文件夹 results1）
    os.makedirs("results1", exist_ok=True)

    # 文件名包含：attack_type, data_split, n_mal, run_id
    filename = f"results1/{args.attack_type}_{args.data_split}_mal{args.n_mal}_run{args.run_id}.csv"

    print(f"[Man / Attack1] Saving metrics to: {filename}")

    # 写 CSV：列为 round, accuracy, loss（顺序按你要求）
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