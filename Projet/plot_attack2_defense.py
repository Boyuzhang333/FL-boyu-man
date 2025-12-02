import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Your folders
FOLDERS = ["results_attack2_median", "results_attack2_trimmed"]

def read_csv_last_acc(filepath):
    try:
        df = pd.read_csv(filepath)
        return df["accuracy"].iloc[-1]
    except:
        return None

def collect(split):
    """Collect all results for IID or Non-IID, return average accuracy for 4 malicious clients"""
    results = {0: [], 1: [], 2: [], 3: []}

    for folder in FOLDERS:
        for n_mal in [0,1,2,3]:
            filename = f"{folder}/label_flipping_{folder.split('_')[-1]}_{split}_mal{n_mal}_run0.csv"
            # Fix naming convention
            filename = f"{folder}/label_flipping_{folder.split('_')[-1]}_{split}_mal{n_mal}_run0.csv"

            # median → label_flipping_median...
            # trimmed → label_flipping_trimmed...
            if "median" in folder:
                prefix = "label_flipping_median"
            else:
                prefix = "label_flipping_trimmed"

            filename = f"{folder}/{prefix}_{split}_mal{n_mal}_run0.csv"

            if os.path.exists(filename):
                acc = read_csv_last_acc(filename)
                if acc is not None:
                    results[n_mal].append(acc)

    # Calculate mean and standard deviation
    means = []
    stds = []
    for n in [0,1,2,3]:
        if len(results[n]) > 0:
            means.append(np.mean(results[n]))
            stds.append(np.std(results[n]))
        else:
            means.append(0.0)
            stds.append(0.0)

    return means, stds


def plot_one(split, outname, color):
    means, stds = collect(split)
    x = [0,1,2,3]

    plt.figure(figsize=(10,5))
    plt.errorbar(x, means, yerr=stds, marker="o", markersize=8, linewidth=2,
                 capsize=5, color=color)

    title = "IID Data Distribution (Model Poisoning)" if split=="iid" else "Non-IID Data Distribution (Model Poisoning)"
    plt.title(title, fontsize=16)
    plt.xlabel("Number of Malicious Clients", fontsize=14)
    plt.ylabel("Final Accuracy", fontsize=14)
    plt.ylim(0,1)
    plt.grid(True, alpha=0.3)
    plt.xticks([0,1,2,3])
    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    print(f"Saved → {outname}")
    plt.close()


if __name__ == "__main__":
    print("Generating IID and Non-IID plots...")

    plot_one("iid", "attack2_model_poisoning_iid.png", "#1f77b4")
    plot_one("non_iid_class", "attack2_model_poisoning_noniid.png", "#ff7f0e")

    print("Done!")
