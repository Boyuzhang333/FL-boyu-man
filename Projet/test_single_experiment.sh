#!/bin/bash


echo "🧪 Testing single experiment with correct parameters..."


source $(conda info --base)/etc/profile.d/conda.sh
conda activate fl-miage


data_split="iid"
n_mal=0
run_id=2

echo "📋 Test configuration:"
echo "   Data Split: $data_split"
echo "   Malicious Clients: $n_mal"
echo "   Run ID: $run_id"


python serveur_attack1.py \
    --round 2 \
    --data_split "$data_split" \
    --n_mal "$n_mal" \
    --run_id "$run_id" &

server_pid=$!


echo "   Waiting for server to start..."
sleep 10


echo "   Starting clients with correct parameters..."

client_pids=()


for ((i=0; i<5; i++)); do
    echo "   Starting client $i..."
    python client.py --node_id $i --data_split "$data_split" &
    client_pids+=($!)
done

echo "   All clients started, waiting for training to complete..."


wait $server_pid
server_exit_code=$?


for pid in "${client_pids[@]}"; do
    if kill -0 $pid 2>/dev/null; then
        kill $pid 2>/dev/null
    fi
done


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
