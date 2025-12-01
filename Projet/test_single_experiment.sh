#!/bin/bash
# 测试单个实验以验证参数修复

echo "🧪 Testing single experiment with correct parameters..."

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fl-miage

# 测试参数
data_split="iid"
n_mal=0
run_id=2

echo "📋 Test configuration:"
echo "   Data Split: $data_split"
echo "   Malicious Clients: $n_mal"
echo "   Run ID: $run_id"

# 启动服务器（后台运行）
python serveur_attack1.py \
    --round 2 \
    --data_split "$data_split" \
    --n_mal "$n_mal" \
    --run_id "$run_id" &

server_pid=$!

# 等待服务器启动
echo "   Waiting for server to start..."
sleep 10

# 启动客户端
echo "   Starting clients with correct parameters..."

client_pids=()

# 启动5个正常客户端 (n_mal=0)
for ((i=0; i<5; i++)); do
    echo "   Starting client $i..."
    python client.py --node_id $i --data_split "$data_split" &
    client_pids+=($!)
done

echo "   All clients started, waiting for training to complete..."

# 等待服务器完成
wait $server_pid
server_exit_code=$?

# 终止所有客户端
for pid in "${client_pids[@]}"; do
    if kill -0 $pid 2>/dev/null; then
        kill $pid 2>/dev/null
    fi
done

# 检查结果
result_file="results1/label_flipping_${data_split}_mal${n_mal}_run${run_id}.csv"

if [ $server_exit_code -eq 0 ] && [ -f "$result_file" ]; then
    echo "   ✅ Test experiment completed successfully!"
    echo "   Result saved to: $result_file"
else
    echo "   ❌ Test experiment failed!"
    echo "   Server exit code: $server_exit_code"
    echo "   Result file exists: $([ -f "$result_file" ] && echo "Yes" || echo "No")"
fi

echo "🧪 Test completed!"
