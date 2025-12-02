#!/usr/bin/env python3
"""
Attack1 Defense Results Visualization
Generate defense plots for Attack1 (Label Flipping) with FedMedian and FedTrimmedAvg
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

def collect_defense_results(defense_name):
    """Collect defense results for a specific defense method"""
    results = {
        'iid': {0: None, 1: None, 2: None, 3: None},
        'non_iid_class': {0: None, 1: None, 2: None, 3: None}
    }
    
    # Results folder based on defense type
    folder = f"results_attack1_{defense_name}"
    
    # Traverse all possible files
    for data_split in ['iid', 'non_iid_class']:
        for n_mal in [0, 1, 2, 3]:
            filename = f"{folder}/label_flipping_{defense_name}_{data_split}_mal{n_mal}_run0.csv"
            if os.path.exists(filename):
                accuracy = read_final_accuracy(filename)
                if accuracy is not None:
                    results[data_split][n_mal] = accuracy
            else:
                print(f"Warning: File not found: {filename}")
    
    return results

def plot_defense_results(defense_name):
    """Plot defense effectiveness graphs"""
    results = collect_defense_results(defense_name)
    
    # Set plot style
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    malicious_clients = [0, 1, 2, 3]
    
    # Plot 1: IID case
    iid_accuracies = []
    
    for n_mal in malicious_clients:
        accuracy = results['iid'][n_mal]
        if accuracy is not None:
            iid_accuracies.append(accuracy)
        else:
            iid_accuracies.append(0)  # Default if missing
    
    ax1.plot(malicious_clients, iid_accuracies, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Malicious Clients')
    ax1.set_ylabel('Final Accuracy')
    ax1.set_title('IID Data Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(malicious_clients)
    
    # Plot 2: Non-IID case
    non_iid_accuracies = []
    
    for n_mal in malicious_clients:
        accuracy = results['non_iid_class'][n_mal]
        if accuracy is not None:
            non_iid_accuracies.append(accuracy)
        else:
            non_iid_accuracies.append(0)  # Default if missing
    
    ax2.plot(malicious_clients, non_iid_accuracies, marker='o', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Number of Malicious Clients')
    ax2.set_ylabel('Final Accuracy')
    ax2.set_title('Non-IID Data Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(malicious_clients)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f'attack1_{defense_name}_iid_vs_non_iid.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Display statistical information
    print(f"\n=== Attack1 (Label Flipping) Results with {defense_name.upper()} Defense ===")
    print("\nIID Data Distribution:")
    for i, n_mal in enumerate(malicious_clients):
        print(f"  {n_mal} malicious clients: {iid_accuracies[i]:.4f}")
    
    print("\nNon-IID Data Distribution:")
    for i, n_mal in enumerate(malicious_clients):
        print(f"  {n_mal} malicious clients: {non_iid_accuracies[i]:.4f}")
    
    plt.show()

def check_defense_folders():
    """Check if defense result folders exist"""
    missing_folders = []
    
    for defense in ['median', 'trimmed']:
        folder = f'results_attack1_{defense}'
        if not os.path.exists(folder):
            missing_folders.append(folder)
    
    if missing_folders:
        print(f"Error: Missing folders: {missing_folders}")
        return False
    
    return True

def count_defense_files():
    """Count CSV files in defense folders"""
    for defense in ['median', 'trimmed']:
        folder = f'results_attack1_{defense}'
        if os.path.exists(folder):
            csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
            print(f"Found {len(csv_files)} CSV files in {folder}/")

if __name__ == "__main__":
    # Check if defense result folders exist
    if not check_defense_folders():
        exit(1)
    
    # Count CSV files
    count_defense_files()
    
    # Generate plots for both defense methods
    for defense in ['median', 'trimmed']:
        print(f"\n{'='*50}")
        print(f"Generating plot for {defense.upper()} defense...")
        print('='*50)
        plot_defense_results(defense)
        print()
