#!/bin/bash
# Automated script to run Attack2 (model_poisoning) under two defenses:
# FedMedian and FedTrimmedAvg
# Covers all combinations of:
#   - data_split: iid / non_iid_class
#   - n_mal: 0–3
#   - run_id: 0–4
#   - defense: median / trimmed

echo "🚀 Starting FULL Attack2 + Defense Experiments"
echo "============================================================="
echo ""

# -------------------------
# 检查 conda 环境
# -------------------------
if ! conda info --envs | grep -q "fl-miage"; then
    echo "❌ Error: conda env fl-miage NOT found!"
    exit 1
fi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate fl-miage

# -------------------------
# 检查文件是否存在
# -------------------------
if [ ! -f "serveur_attack2_defense.py" ]; then
    echo "❌ Error: serveur_attack2_defense.py NOT found!"
    exit 1
fi

echo "📌 Using serveur_attack2_defense.py to run experiments"
echo ""

# -------------------------
# 配置参数
# -------------------------
data_splits=("iid" "non_iid_class")
malicious_counts=(0 1 2 3)
runs=(0 1 2 3 4)

defenses=("median" "trimmed")
folders=("results_attack2_median" "results_attack2_trimmed")

mkdir -p results_attack2_median
mkdir -p results_attack2_trimmed

total=$((2 * 4 * 5 * 2))    # 80 次实验
current=1

start_time=$(date)
echo "🕒 Start: $start_time"
echo ""

# -------------------------
# 主循环
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

                outfile="$outdir/model_poisoning_${defense}_${split}_mal${n_mal}_run${run_id}.csv"

                if [ -f "$outfile" ]; then
                    echo "⏭️  [$current/$total] Skip existing: $outfile"
                    ((current++))
                    continue
                fi

                echo "🔄 [$current/$total] Running:"
                echo "   Defense: $defense"
                echo "   Split: $split"
                echo "   Malicious: $n_mal"
                echo "   Run: $run_id"

                # -------------------------
                # 启动服务器（后台）
                # -------------------------
                python serveur_attack2_defense.py \
                    --round 20 \
                    --data_split "$split" \
                    --attack_type model_poisoning \
                    --defense "$defense" \
                    --n_mal "$n_mal" \
                    --run_id "$run_id" &

                server_pid=$!
                sleep 10

                # -------------------------
                # 启动客户端
                # -------------------------
                client_pids=()

                # 恶意客户端
                for ((j=0; j<n_mal; j++)); do
                    python client_mal.py \
                        --node_id $j \
                        --data_split "$split" \
                        --attack_type model_poisoning &
                    client_pids+=($!)
                done

                # 正常客户端
                for ((j=n_mal; j<5; j++)); do
                    python client.py \
                        --node_id $j \
                        --data_split "$split" &
                    client_pids+=($!)
                done

                # -------------------------
                # 等待服务器完成
                # -------------------------
                wait $server_pid
                exit_code=$?

                # 杀掉残留的客户端
                for pid in "${client_pids[@]}"; do
                    kill $pid 2>/dev/null
                done

                # -------------------------
                # 移动结果文件
                # -------------------------
                internal="results_attack2/model_poisoning_${defense}_${split}_mal${n_mal}_run${run_id}.csv"

                if [ $exit_code -eq 0 ] && [ -f "$internal" ]; then
                    mv "$internal" "$outfile"
                    echo "   ✅ Saved: $outfile"
                else
                    echo "   ❌ Failed. File missing: $internal"
                fi

                echo ""
                ((current++))
                sleep 5

            done
        done
    done
done

# -------------------------
# 完成输出
# -------------------------
echo "============================================================="
echo "🏁 All Attack2 defense experiments completed!"
echo "🕒 Start: $start_time"
echo "🕒 End: $(date)"
echo "============================================================="

echo "📊 Summary:"
echo "  Results saved to:"
echo "  - results_attack2_median/"
echo "  - results_attack2_trimmed/"
