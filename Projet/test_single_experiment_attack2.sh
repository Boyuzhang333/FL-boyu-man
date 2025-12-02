#!/bin/bash
# Test single experiment (Attack2: model_poisoning) to verify parameter fixes

echo "🧪 Testing single experiment (Attack2: model_poisoning) with correct parameters..."

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fl-miage

# Test parameters
data_split="iid"
n_mal=0
run_id=2

echo "📋 Test configuration:"
echo "   Data Split: $data_split"
echo "   Malicious Clients: $n_mal"
echo "   Run ID: $run_id"

echo "   Starting serveur_attack2.py..."
# Start server (background)
python serveur_attack2.py \
    --round 2 \
    --data_split "$data_split" \
    --n_mal "$n_mal" \
    --run_id "$run_id" \
    --attack_type model_poisoning &

server_pid=$!

# Wait for server to start
echo "   Waiting for server to start..."
sleep 10

# Start clients
echo "   Starting clients with model_poisoning attack..."

client_pids=()

for ((i=0; i<5; i++)); do
    echo "   Starting client $i..."
    python client_mal.py --node_id $i --data_split "$data_split" --attack_type model_poisoning &
    client_pids+=($!)
done

echo "   All clients started, waiting for training to complete..."

# Wait for server to complete
wait $server_pid
server_exit_code=$?

# Terminate all clients
for pid in "${client_pids[@]}"; do
    if kill -0 $pid 2>/dev/null; then
        kill $pid 2>/dev/null
    fi
done

# Check results
result_file="results2/model_poisoning_${data_split}_mal${n_mal}_run${run_id}.csv"

if [ $server_exit_code -eq 0 ] && [ -f "$result_file" ]; then
    echo "   ✅ Test experiment completed successfully!"
    echo "   Result saved to: $result_file"
else
    echo "   ❌ Test experiment failed!"
    echo "   Server exit code: $server_exit_code"
    echo "   Result file exists: $([ -f "$result_file" ] && echo "Yes" || echo "No")"
fi

echo "🧪 Test completed!"
