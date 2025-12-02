#!/bin/bash
# Attack1 (Label Flipping) Experiment Guide
# Label flipping attack experiment guide

echo "🎯 Attack1 Label Flipping Experiment Guide"
echo "================================================"
echo ""

echo "📋 Usage of serveur_attack1.py:"
echo "python serveur_attack1.py [options]"
echo ""

echo "🔧 Available Options:"
echo "  --round N          Number of training rounds (default: 10)"
echo "  --data_split TYPE  Data distribution: 'iid' or 'non_iid_class' (default: iid)"
echo "  --attack_type STR  Attack type (default: label_flipping)"
echo "  --n_mal N          Number of malicious clients (default: 0)"
echo "  --run_id N         Run ID for experiment repetition (default: 0)"
echo ""

echo "📁 Output:"
echo "  - CSV files saved in: results1/"
echo "  - Filename format: {attack_type}_{data_split}_mal{n_mal}_run{run_id}.csv"
echo ""

echo "💡 Example Commands:"
echo ""

echo "# Experiment 1: 0 malicious clients, iid data"
echo "python serveur_attack1.py --round 20 --data_split iid --n_mal 0 --run_id 0"
echo ""

echo "# Experiment 2: 1 malicious client, iid data"  
echo "python serveur_attack1.py --round 20 --data_split iid --n_mal 1 --run_id 0"
echo ""

echo "# Experiment 3: 2 malicious clients, iid data"
echo "python serveur_attack1.py --round 20 --data_split iid --n_mal 2 --run_id 0"
echo ""

echo "# Experiment 4: 3 malicious clients, iid data"
echo "python serveur_attack1.py --round 20 --data_split iid --n_mal 3 --run_id 0"
echo ""

echo "# Experiment 5: 0 malicious clients, non-iid data"
echo "python serveur_attack1.py --round 20 --data_split non_iid_class --n_mal 0 --run_id 0"
echo ""

echo "# Experiment 6: 1 malicious client, non-iid data"
echo "python serveur_attack1.py --round 20 --data_split non_iid_class --n_mal 1 --run_id 0"
echo ""

echo "# Experiment 7: 2 malicious clients, non-iid data"
echo "python serveur_attack1.py --round 20 --data_split non_iid_class --n_mal 2 --run_id 0"
echo ""

echo "# Experiment 8: 3 malicious clients, non-iid data"
echo "python serveur_attack1.py --round 20 --data_split non_iid_class --n_mal 3 --run_id 0"
echo ""

echo "🔄 For Multiple Runs (Repetitions):"
echo "# Run 1"
echo "python serveur_attack1.py --round 20 --data_split iid --n_mal 1 --run_id 0"
echo "# Run 2"  
echo "python serveur_attack1.py --round 20 --data_split iid --n_mal 1 --run_id 1"
echo "# Run 3"
echo "python serveur_attack1.py --round 20 --data_split iid --n_mal 1 --run_id 2"
echo ""

echo "⚠️  Important Notes:"
echo "  1. Start serveur_attack1.py first (server)"
echo "  2. Then start clients (mix of honest and malicious)"
echo "  3. Results automatically saved when server stops"
echo "  4. Each experiment creates separate CSV file"
echo "  5. File names include all experiment parameters"
echo ""

echo "🖥️  Client Commands (run after starting server):"
echo "# For N malicious clients, use N instances of client_mal.py with label_flipping"
echo "# For remaining honest clients, use client.py"
echo ""
echo "# Example for 1 malicious client + 4 honest clients:"
echo "# Terminal 1: Server"
echo "python serveur_attack1.py --round 20 --data_split iid --n_mal 1 --run_id 0"
echo ""
echo "# Terminal 2: Malicious client"
echo "python client_mal.py --node_id 0 --n 5 --data_split iid --attack_type label_flipping"
echo ""
echo "# Terminal 3-6: Honest clients"
echo "python client.py --node_id 1 --n 5 --data_split iid"
echo "python client.py --node_id 2 --n 5 --data_split iid"
echo "python client.py --node_id 3 --n 5 --data_split iid"
echo "python client.py --node_id 4 --n 5 --data_split iid"
echo ""

echo "📊 Expected Output Files:"
echo "# For the above example:"
echo "results1/label_flipping_iid_mal1_run0.csv"
