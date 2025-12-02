#!/bin/bash
# Automated script to run Attack1 label_flipping with FedMedian & FedTrimmedAvg
# 批量运行 IID + NonIID × mal0–3 × run0–4 × 两个防御策略

echo "🚀 Starting FULL Attack1 + Defense Experiments"
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
if [ ! -f "serveur_attack1_defense.py" ]; then
    echo "❌ Error: serveur_attack1_defense.py NOT found!"
    exit 1
fi

echo "📌 Using serveur_attack1_defense.py to run experiments"
echo ""

# -------------------------
# 配置参数
# -------------------------
data_splits=("iid" "non_iid_class")
malicious_counts=(0 1 2 3)
runs=(0 1 2 3 4)
defenses=("median" "trimmed")
folders=("results_attack1_median" "results_attack1_trimmed")

mkdir -p results_attack1_median
mkdir -p results_attack1_trimmed

total=$((2 * 4 * 5 * 2))    # 80 次
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

                # 输出文件名
                outfile="$outdir/label_flipping_${defense}_${split}_mal${n_mal}_run${run_id}.csv"

                # 如果已完成则跳过
                if [ -f "$outfile" ]; then
                    echo "⏭️  [$current/$total] Skip existing $outfile"
                    ((current++))
                    continue
                fi

                echo "🔄 [$current/$total] Running experiment:"
                echo "   Defense: $defense"
                echo "   Split: $split"
                echo "   Malicious: $n_mal"
                echo "   Run: $run_id"

                # -------------------------
                # 启动服务器
                # -------------------------
                python serveur_attack1_defense.py \
                    --round 20 \
                    --data_split "$split" \
                    --attack_type label_flipping \
                    --defense "$defense" \
                    --n_mal "$n_mal" \
                    --run_id "$run_id" &

                server_pid=$!
                sleep 10

                # 启动客户端
                client_pids=()

                # 恶意客户端
                for ((j=0; j<n_mal; j++)); do
                    python client_mal.py \
                        --node_id $j \
                        --data_split "$split" \
                        --attack_type label_flipping &

                    client_pids+=($!)
                done

                # 正常客户端
                for ((j=n_mal; j<5; j++)); do
                    python client.py \
                        --node_id $j \
                        --data_split "$split" &
                    client_pids+=($!)
                done

                # 等待服务器结束
                wait $server_pid
                exit_code=$?

                # 杀死所有客户端
                for pid in "${client_pids[@]}"; do
                    kill $pid 2>/dev/null
                done

                # 服务器保存的是 results_attack1/，需要移动
                internal_file="results_attack1/label_flipping_${defense}_${split}_mal${n_mal}_run${run_id}.csv"

                if [ $exit_code -eq 0 ] && [ -f "$internal_file" ]; then
                    mv "$internal_file" "$outfile"
                    echo "   ✅ Saved: $outfile"
                else
                    echo "   ❌ Failed. File not found!"
                fi

                echo ""
                ((current++))
                sleep 5

            done
        done
    done
done

echo "============================================================="
echo "🏁 All experiments completed!"
echo "🕒 Start: $start_time"
echo "🕒 End: $(date)"
echo "============================================================="

echo "📊 Summary:"
echo "  Results saved into:"
echo "  - results_attack1_median/"
echo "  - results_attack1_trimmed/"
