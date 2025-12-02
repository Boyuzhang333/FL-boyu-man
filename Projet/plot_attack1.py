#!/usr/bin/env python3
"""
Attack1 (Label Flipping) Results Visualization
Generate two plots: Attack effects under IID and Non-IID conditions
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def read_final_accuracy(filepath):
    """Read the final accuracy value from CSV file"""
    try:
        df = pd.read_csv(filepath)
        return df['accuracy'].iloc[-1]
    except:
        print(f"Warning: Cannot read {filepath}")
        return None

def collect_results():
    """Collect all experiment results"""
    results = {
        'iid': {0: [], 1: [], 2: [], 3: []},
        'non_iid_class': {0: [], 1: [], 2: [], 3: []}
    }
    
    # Traverse all possible files
    for data_split in ['iid', 'non_iid_class']:
        for n_mal in [0, 1, 2, 3]:
            for run_id in range(5):  # 5 repetitions
                filename = f"results1/label_flipping_{data_split}_mal{n_mal}_run{run_id}.csv"
                if os.path.exists(filename):
                    accuracy = read_final_accuracy(filename)
                    if accuracy is not None:
                        results[data_split][n_mal].append(accuracy)
    
    return results

def plot_results():
    """Plot attack effectiveness graphs"""
    results = collect_results()
    
    # Set plot style
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    malicious_clients = [0, 1, 2, 3]
    
    # Plot 1: IID case
    iid_means = []
    iid_stds = []
    
    for n_mal in malicious_clients:
        accuracies = results['iid'][n_mal]
        if accuracies:
            iid_means.append(np.mean(accuracies))
            iid_stds.append(np.std(accuracies))
        else:
            iid_means.append(0)
            iid_stds.append(0)
    
    ax1.errorbar(malicious_clients, iid_means, yerr=iid_stds, 
                marker='o', linewidth=2, markersize=8, capsize=5)
    ax1.set_xlabel('Number of Malicious Clients')
    ax1.set_ylabel('Final Accuracy')
    ax1.set_title('IID Data Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(malicious_clients)
    
    # Plot 2: Non-IID case
    non_iid_means = []
    non_iid_stds = []
    
    for n_mal in malicious_clients:
        accuracies = results['non_iid_class'][n_mal]
        if accuracies:
            non_iid_means.append(np.mean(accuracies))
            non_iid_stds.append(np.std(accuracies))
        else:
            non_iid_means.append(0)
            non_iid_stds.append(0)
    
    ax2.errorbar(malicious_clients, non_iid_means, yerr=non_iid_stds,
                marker='o', linewidth=2, markersize=8, capsize=5, color='orange')
    ax2.set_xlabel('Number of Malicious Clients')
    ax2.set_ylabel('Final Accuracy')
    ax2.set_title('Non-IID Data Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(malicious_clients)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'attack1_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Display statistical information
    print("\n=== Attack1 (Label Flipping) Results ===")
    print("\nIID Data Distribution:")
    for i, n_mal in enumerate(malicious_clients):
        print(f"  {n_mal} malicious clients: {iid_means[i]:.4f} ± {iid_stds[i]:.4f}")
    
    print("\nNon-IID Data Distribution:")
    for i, n_mal in enumerate(malicious_clients):
        print(f"  {n_mal} malicious clients: {non_iid_means[i]:.4f} ± {non_iid_stds[i]:.4f}")
    
    plt.show()

if __name__ == "__main__":
    # Check if results1 directory exists
    if not os.path.exists('results1'):
        print("Error: results1/ directory not found!")
        exit(1)
    
    # Check number of CSV files
    csv_files = [f for f in os.listdir('results1') if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in results1/")
    
    # Generate plots
    plot_results()
