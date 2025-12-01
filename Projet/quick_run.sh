#!/bin/bash
# Quick batch script to run specific experiments
# 快速批量运行特定实验脚本

echo "🎯 Quick Experiment Runner"
echo "=========================="
echo ""

# 检查参数
if [ $# -lt 3 ]; then
    echo "Usage: $0 <data_split> <n_mal> <start_run_id> [end_run_id]"
    echo ""
    echo "Examples:"
    echo "  $0 iid 0 2           # Run iid_mal0_run2"
    echo "  $0 iid 2 0 4         # Run iid_mal2_run0 through run4"
    echo "  $0 non_iid_class 1 0 4  # Run non_iid_class_mal1_run0 through run4"
    exit 1
fi

data_split=$1
n_mal=$2
start_run=$3
end_run=${4:-$start_run}  # 如果没有提供end_run，就只运行一个

# 激活环境
echo "🔄 Activating fl-miage environment..."
eval "$(conda shell.bash hook)"
conda activate fl-miage

echo "📋 Running experiments:"
echo "  Data Split: $data_split"
echo "  Malicious Clients: $n_mal"
echo "  Run IDs: $start_run to $end_run"
echo ""

for ((run_id=start_run; run_id<=end_run; run_id++)); do
    result_file="results1/label_flipping_${data_split}_mal${n_mal}_run${run_id}.csv"
    
    if [ -f "$result_file" ]; then
        echo "⏭️  Skipping run $run_id (already exists)"
        continue
    fi
    
    echo "🔄 Running experiment: ${data_split}_mal${n_mal}_run${run_id}"
    
    # 启动服务器（后台）
    python serveur_attack1.py \
        --round 20 \
        --data_split "$data_split" \
        --n_mal "$n_mal" \
        --run_id "$run_id" &
    
    server_pid=$!
    
    # 等待服务器启动
    sleep 8
    
    # 启动客户端
    client_pids=()
    
    # 恶意客户端
    for ((i=0; i<n_mal; i++)); do
        python client_mal.py --cid $i --data_split "$data_split" &
        client_pids+=($!)
    done
    
    # 正常客户端
    for ((i=n_mal; i<5; i++)); do
        python client.py --cid $i --data_split "$data_split" &
        client_pids+=($!)
    done
    
    echo "   Training in progress..."
    
    # 等待服务器完成
    wait $server_pid
    
    # 清理客户端进程
    for pid in "${client_pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null
        fi
    done
    
    # 检查结果
    if [ -f "$result_file" ]; then
        echo "   ✅ Completed: $result_file"
    else
        echo "   ❌ Failed: $result_file not generated"
    fi
    
    echo ""
    sleep 3
done

echo "✨ Batch completed!"
