#!/usr/bin/env python3
"""
Attack2 (Model Poisoning) Results Visualization | 攻击2（模型中毒）结果可视化
Generate two plots: Attack effects under IID and Non-IID conditions | 生成两张图：IID和Non-IID情况下的攻击效果
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def read_final_accuracy(filepath):
    """Read the final accuracy value from CSV file | 读取CSV文件的最后一行accuracy值"""
    try:
        df = pd.read_csv(filepath)
        return df['accuracy'].iloc[-1]
    except Exception as e:
        print(f"Warning: Cannot read {filepath}: {e}")
        return None

def collect_results():
    """Collect all experiment results | 收集所有实验结果"""
    results = {
        'iid': {0: [], 1: [], 2: [], 3: []},
        'non_iid_class': {0: [], 1: [], 2: [], 3: []}
    }
    
    # 遍历你已有的所有配置
    for data_split in ['iid', 'non_iid_class']:
        for n_mal in [0, 1, 2, 3]:
            for run_id in range(5):  # 5 次重复
                filename = f"results2/model_poisoning_{data_split}_mal{n_mal}_run{run_id}.csv"
                if os.path.exists(filename):
                    accuracy = read_final_accuracy(filename)
                    if accuracy is not None:
                        results[data_split][n_mal].append(accuracy)
                else:
                    # 可选：调试用，看看哪些文件缺失
                    # print(f"Missing file: {filename}")
                    pass
    
    return results

def plot_results():
    """Plot attack effectiveness graphs | 绘制攻击效果图"""
    results = collect_results()
    
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    malicious_clients = [0, 1, 2, 3]
    
    # IID
    iid_means, iid_stds = [], []
    for n_mal in malicious_clients:
        accs = results['iid'][n_mal]
        if accs:
            iid_means.append(np.mean(accs))
            iid_stds.append(np.std(accs))
        else:
            iid_means.append(0.0)
            iid_stds.append(0.0)
    
    ax1.errorbar(malicious_clients, iid_means, yerr=iid_stds,
                 marker='o', linewidth=2, markersize=8, capsize=5)
    ax1.set_xlabel('Number of Malicious Clients')
    ax1.set_ylabel('Final Accuracy')
    ax1.set_title('IID Data Distribution (Model Poisoning)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(malicious_clients)
    
    # Non-IID
    non_means, non_stds = [], []
    for n_mal in malicious_clients:
        accs = results['non_iid_class'][n_mal]
        if accs:
            non_means.append(np.mean(accs))
            non_stds.append(np.std(accs))
        else:
            non_means.append(0.0)
            non_stds.append(0.0)
    
    ax2.errorbar(malicious_clients, non_means, yerr=non_stds,
                 marker='o', linewidth=2, markersize=8, capsize=5, color='orange')
    ax2.set_xlabel('Number of Malicious Clients')
    ax2.set_ylabel('Final Accuracy')
    ax2.set_title('Non-IID Data Distribution (Model Poisoning)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(malicious_clients)
    
    plt.tight_layout()
    output_file = 'attack2_model_poisoning_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # 打印统计结果，方便你写报告
    print("\n=== Attack2 (Model Poisoning) Results ===")
    print("\nIID Data Distribution:")
    for i, n_mal in enumerate(malicious_clients):
        print(f"  {n_mal} malicious clients: {iid_means[i]:.4f} ± {iid_stds[i]:.4f}")
    
    print("\nNon-IID Data Distribution:")
    for i, n_mal in enumerate(malicious_clients):
        print(f"  {n_mal} malicious clients: {non_means[i]:.4f} ± {non_stds[i]:.4f}")
    
    plt.show()

if __name__ == "__main__":
    if not os.path.exists('results2'):
        print("Error: results2/ directory not found!")
        exit(1)
    
    csv_files = [f for f in os.listdir('results2') if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in results2/")
    
    plot_results()
