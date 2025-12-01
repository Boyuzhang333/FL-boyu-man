#!/bin/bash
# Automated script to run remaining Attack1 label flipping experiments
# 自动化运行剩余的Attack1标签翻转攻击实验

echo "🚀 Starting Automated Attack1 Experiments"
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
if [ ! -f "serveur_attack1.py" ]; then
    echo "❌ Error: serveur_attack1.py not found in current directory!"
    echo "Please run this script from the Projet/ directory"
    exit 1
fi

# 定义实验配置
echo "📋 Experiment Configuration:"
echo "- 8 configurations × 5 repetitions = 40 total experiments"
echo "- Already completed: 3 experiments"
echo "- Remaining: 37 experiments"
echo ""

# 定义需要完成的实验
experiments=(
    # IID experiments - complete the repetitions
    "iid 0 2"  # Complete runs 2-4 for 0 malicious clients
    "iid 0 3"
    "iid 0 4"
    "iid 1 1"  # Complete runs 1-4 for 1 malicious client
    "iid 1 2"
    "iid 1 3"
    "iid 1 4"
    "iid 2 0"  # All runs for 2 malicious clients
    "iid 2 1"
    "iid 2 2"
    "iid 2 3"
    "iid 2 4"
    "iid 3 0"  # All runs for 3 malicious clients
    "iid 3 1"
    "iid 3 2"
    "iid 3 3"
    "iid 3 4"
    
    # Non-IID experiments - all runs for all configurations
    "non_iid_class 0 0"  # 0 malicious clients
    "non_iid_class 0 1"
    "non_iid_class 0 2"
    "non_iid_class 0 3"
    "non_iid_class 0 4"
    "non_iid_class 1 0"  # 1 malicious client
    "non_iid_class 1 1"
    "non_iid_class 1 2"
    "non_iid_class 1 3"
    "non_iid_class 1 4"
    "non_iid_class 2 0"  # 2 malicious clients
    "non_iid_class 2 1"
    "non_iid_class 2 2"
    "non_iid_class 2 3"
    "non_iid_class 2 4"
    "non_iid_class 3 0"  # 3 malicious clients
    "non_iid_class 3 1"
    "non_iid_class 3 2"
    "non_iid_class 3 3"
    "non_iid_class 3 4"
)

total_experiments=${#experiments[@]}
current_experiment=1

echo "⏳ Starting experiments... This will take a while (estimated ${total_experiments} × 15 minutes = $((total_experiments * 15 / 60)) hours)"
echo ""

# 记录开始时间
start_time=$(date)
echo "🕐 Start time: $start_time"
echo ""

# 运行所有实验
for exp in "${experiments[@]}"; do
    # 解析参数
    read -r data_split n_mal run_id <<< "$exp"
    
    # 生成文件名检查是否已存在
    result_file="results1/label_flipping_${data_split}_mal${n_mal}_run${run_id}.csv"
    
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
    
    # 启动服务器（后台运行）
    python serveur_attack1.py \
        --round 20 \
        --data_split "$data_split" \
        --n_mal "$n_mal" \
        --run_id "$run_id" &
    
    server_pid=$!
    
    # 等待服务器启动
    echo "   Waiting for server to start..."
    sleep 10
    
    # 启动客户端
    echo "   Starting clients..."
    
    # 根据恶意客户端数量启动相应的客户端
    client_pids=()
    
    # 启动恶意客户端
    for ((i=0; i<n_mal; i++)); do
        python client_mal.py --node_id $i --data_split "$data_split" &
        client_pids+=($!)
    done
    
    # 启动正常客户端
    remaining_clients=$((5 - n_mal))
    for ((i=n_mal; i<5; i++)); do
        python client.py --node_id $i --data_split "$data_split" &
        client_pids+=($!)
    done
    
    echo "   Training in progress... (estimated 15 minutes)"
    
    # 等待服务器完成
    wait $server_pid
    server_exit_code=$?
    
    # 终止所有客户端
    for pid in "${client_pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null
        fi
    done
    
    # 检查实验是否成功
    if [ $server_exit_code -eq 0 ] && [ -f "$result_file" ]; then
        echo "   ✅ Experiment completed successfully!"
    else
        echo "   ❌ Experiment failed!"
        echo "   Server exit code: $server_exit_code"
        echo "   Result file exists: $([ -f "$result_file" ] && echo "Yes" || echo "No")"
    fi
    
    echo ""
    ((current_experiment++))
    
    # 短暂休息避免系统过载
    sleep 5
done

# 记录结束时间
end_time=$(date)
echo "🏁 All experiments completed!"
echo "🕐 Start time: $start_time"
echo "🕐 End time: $end_time"
echo ""

# 显示结果统计
echo "📊 Results Summary:"
echo "=================="
total_files=$(ls results1/*.csv 2>/dev/null | wc -l)
echo "Total CSV files: $total_files"
echo ""

echo "📁 Results by configuration:"
for data_split in "iid" "non_iid_class"; do
    echo "  $data_split:"
    for n_mal in 0 1 2 3; do
        count=$(ls results1/label_flipping_${data_split}_mal${n_mal}_run*.csv 2>/dev/null | wc -l)
        echo "    $n_mal malicious clients: $count/5 runs completed"
    done
done

echo ""
echo "✨ Script execution completed!"
echo "Next step: Run data analysis and visualization scripts"
