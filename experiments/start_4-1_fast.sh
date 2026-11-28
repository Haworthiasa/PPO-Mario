#!/usr/bin/env bash
# Bắt đầu lượt 4-1 "ít bước" (docs/optimize-1-1.md, Mục 13): cấu hình c1-e0-f1-g1, 819.200 bước, seed 1. Kiểm luật
# của đề trước (experiments/check_rules.sh); có điều sai thì không chạy.
#   bash experiments/start_4-1_fast.sh        (trong WSL, từ thư mục gốc của kho)
set -eu
cd "$(dirname "$0")/.."
DIR=runs/4-1/tuning/fast-819k
if [ -f "$DIR.pid" ] && kill -0 "$(cat "$DIR.pid")" 2>/dev/null; then echo "lượt 4-1 ít bước đang chạy"; exit 0; fi
bash experiments/check_rules.sh > /dev/null || { echo "kiểm luật báo sai; không chạy"; exit 1; }
OVER=$("$HOME/.venvs/mario-ppo/bin/python" -c \
  "import sys; sys.path.insert(0, 'experiments'); from tuning2_report import overrides; print(overrides('c1-e0-f1-g1'))")
mkdir -p runs/4-1/tuning
echo "$(date '+%F %T') khóa ghi đè: $OVER" >> "$DIR.log"
OVER="$OVER" setsid -f nohup bash experiments/run_single_arm.sh 4-1 "$DIR" configs/group-4-1-tuning.yaml 819200 \
  "seed=1" >> "$DIR.log" 2>&1 < /dev/null
sleep 3
echo "đã bắt đầu $DIR, nhóm $(cat "$DIR.pid")"
