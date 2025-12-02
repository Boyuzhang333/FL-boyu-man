#!/bin/bash
# Automated script to run remaining Attack2 model poisoning experiments
# 自动化运行剩余的Attack2梯度上升攻击实验

echo "🚀 Starting Automated Attack2 Experiments"
echo "=========================================="
echo ""

# 检查环境
if ! conda info --envs | grep -q "fl-miage"; then
    echo "❌ Error: fl-miage conda environment not found!"
    echo "Please create the environment first:"
    echo "conda env create -f environment.yml"
    exit 1
fi

# 激活环境
echo "🔄 Activating fl-miage environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fl-miage

# 检查当前目录
if [ ! -f "serveur_attack2.py" ]; then
    echo "❌ Error: serveur_attack2.py not found in current directory!"
    echo "Please run this script from the Projet/ directory"
    exit 1
fi

# 定义实验配置
echo "📋 Experiment Configuration:"
echo "- 8 configurations × 5 repetitions = 40 total experiments"
echo ""

# 定义需要完成的实验
experiments=(
    # IID experiments
    "iid 0 0"
    "iid 0 1"
    "iid 0 2"
    "iid 0 3"
    "iid 0 4"
    "iid 1 0"
    "iid 1 1"
    "iid 1 2"
    "iid 1 3"
    "iid 1 4"
    "iid 2 0"
    "iid 2 1"
    "iid 2 2"
    "iid 2 3"
    "iid 2 4"
    "iid 3 0"
    "iid 3 1"
    "iid 3 2"
    "iid 3 3"
    "iid 3 4"
    # Non-IID experiments
    "non_iid_class 0 0"
    "non_iid_class 0 1"
    "non_iid_class 0 2"
    "non_iid_class 0 3"
    "non_iid_class 0 4"
    "non_iid_class 1 0"
    "non_iid_class 1 1"
    "non_iid_class 1 2"
    "non_iid_class 1 3"
    "non_iid_class 1 4"
    "non_iid_class 2 0"
    "non_iid_class 2 1"
    "non_iid_class 2 2"
    "non_iid_class 2 3"
    "non_iid_class 2 4"
    "non_iid_class 3 0"
    "non_iid_class 3 1"
    "non_iid_class 3 2"
    "non_iid_class 3 3"
    "non_iid_class 3 4"
)

total_experiments=${#experiments[@]}
current_experiment=1

# 记录开始时间
start_time=$(date)
echo "🕐 Start time: $start_time"
echo ""

# 运行所有实验
for exp in "${experiments[@]}"; do
    read -r data_split n_mal run_id <<< "$exp"
    result_file="results2/model_poisoning_${data_split}_mal${n_mal}_run${run_id}.csv"
    if [ -f "$result_file" ]; then
        echo "⏭️  [$current_experiment/$total_experiments] Skipping (already exists): $result_file"
        ((current_experiment++))
        continue
    fi
    echo "🔄 [$current_experiment/$total_experiments] Running experiment:"
    echo "   Data Split: $data_split"
    echo "   Malicious Clients: $n_mal"
    echo "   Run ID: $run_id"
    echo "   Output: $result_file"
    python serveur_attack2.py \
        --round 20 \
        --data_split "$data_split" \
        --n_mal "$n_mal" \
        --run_id "$run_id" \
        --attack_type model_poisoning &
    server_pid=$!
    echo "   Waiting for server to start..."
    sleep 10
    client_pids=()
    for ((i=0; i<n_mal; i++)); do
        python client_mal.py --node_id $i --data_split "$data_split" --attack_type model_poisoning &
        client_pids+=($!)
    done
    remaining_clients=$((5 - n_mal))
    for ((i=n_mal; i<5; i++)); do
        python client.py --node_id $i --data_split "$data_split" &
        client_pids+=($!)
    done
    echo "   Training in progress... (estimated 15 minutes)"
    wait $server_pid
    server_exit_code=$?
    for pid in "${client_pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null
        fi
    done
    if [ $server_exit_code -eq 0 ] && [ -f "$result_file" ]; then
        echo "   ✅ Experiment completed successfully!"
    else
        echo "   ❌ Experiment failed!"
        echo "   Server exit code: $server_exit_code"
        echo "   Result file exists: $([ -f "$result_file" ] && echo "Yes" || echo "No")"
    fi
    echo ""
    ((current_experiment++))
    sleep 5
done

end_time=$(date)
echo "🏁 All experiments completed!"
echo "🕐 Start time: $start_time"
echo "🕐 End time: $end_time"
echo ""

total_files=$(ls results2/*.csv 2>/dev/null | wc -l)
echo "Total CSV files: $total_files"
echo ""
echo "📁 Results by configuration:"
for data_split in "iid" "non_iid_class"; do
    echo "  $data_split:"
    for n_mal in 0 1 2 3; do
        count=$(ls results2/model_poisoning_${data_split}_mal${n_mal}_run*.csv 2>/dev/null | wc -l)
        echo "    $n_mal malicious clients: $count/5 runs completed"
    done
done

echo ""
echo "✨ Script execution completed!"
echo "Next step: Run data analysis and visualization scripts"
