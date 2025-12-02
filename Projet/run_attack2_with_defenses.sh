#!/bin/bash
# Automated script to run attack2 label_flipping with FedMedian & FedTrimmedAvg
# Now run each case only once (total 16 runs)

echo "🚀 Starting FULL attack2 + Defense Experiments"
echo "============================================================="
echo ""

# -------------------------
# Check conda environment
# -------------------------
if ! conda info --envs | grep -q "fl-miage"; then
    echo "❌ Error: conda env fl-miage NOT found!"
    exit 1
fi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate fl-miage

# -------------------------
# Check if files exist
# -------------------------
if [ ! -f "serveur_attack2_defense.py" ]; then
    echo "❌ Error: serveur_attack2_defense.py NOT found!"
    exit 1
fi

echo "📌 Using serveur_attack2_defense.py to run experiments"
echo ""

# -------------------------
# Configuration parameters (run each case only once)
# -------------------------
data_splits=("iid" "non_iid_class")
malicious_counts=(0 1 2 3)
runs=(0)                     # ⭐⭐ ← Modified: only run run_id=0
defenses=("median" "trimmed")
folders=("results_attack2_median" "results_attack2_trimmed")

mkdir -p results_attack2_median
mkdir -p results_attack2_trimmed

total=$((2 * 2 * 4 * 1))    # ⭐ Total runs=16
current=1

start_time=$(date)
echo "🕒 Start: $start_time"
echo ""

# -------------------------
# Main loop
# -------------------------
for i in ${!defenses[@]}; do
    defense=${defenses[$i]}
    outdir=${folders[$i]}

    echo "============================================================="
    echo "🏰 Running Defense: $defense → Output: $outdir/"
    echo "============================================================="

    for split in "${data_splits[@]}"; do
        for n_mal in "${malicious_counts[@]}"; do
            for run_id in "${runs[@]}"; do

                outfile="$outdir/label_flipping_${defense}_${split}_mal${n_mal}_run${run_id}.csv"

                if [ -f "$outfile" ]; then
                    echo "⏭️  [$current/$total] Skip existing $outfile"
                    ((current++))
                    continue
                fi

                echo "🔄 [$current/$total] Running:"
                echo "   Defense: $defense"
                echo "   Split: $split"
                echo "   Malicious: $n_mal"
                echo "   Run: $run_id"

                python serveur_attack2_defense.py \
                    --round 20 \
                    --data_split "$split" \
                    --attack_type label_flipping \
                    --defense "$defense" \
                    --n_mal "$n_mal" \
                    --run_id "$run_id" &

                server_pid=$!
                sleep 10

                client_pids=()

                for ((j=0; j<n_mal; j++)); do
                    python client_mal.py \
                        --node_id $j \
                        --data_split "$split" \
                        --attack_type label_flipping &
                    client_pids+=($!)
                done

                for ((j=n_mal; j<5; j++)); do
                    python client.py \
                        --node_id $j \
                        --data_split "$split" &
                    client_pids+=($!)
                done

                wait $server_pid
                exit_code=$?

                for pid in "${client_pids[@]}"; do
                    kill $pid 2>/dev/null
                done

                internal_file="results_attack2/label_flipping_${defense}_${split}_mal${n_mal}_run${run_id}.csv"

                if [ $exit_code -eq 0 ] && [ -f "$internal_file" ]; then
                    mv "$internal_file" "$outfile"
                    echo "   ✅ Saved: $outfile"
                else
                    echo "   ❌ Failed! File missing!"
                fi

                echo ""
                ((current++))
                sleep 5
            done
        done
    done
done

echo "============================================================="
echo "🏁 All 16 experiments completed!"
echo "🕒 Start: $start_time"
echo "🕒 End: $(date)"
echo "============================================================="
