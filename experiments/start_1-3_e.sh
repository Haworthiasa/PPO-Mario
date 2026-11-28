#!/usr/bin/env bash
# Lượt 1-3 phương án E (docs/optimize-1-1.md, Mục 18.6 và 19): cấu hình c1-e0-f1-g1 như lượt đầu, chỉ đổi hai thứ:
# γ 0,9 -> 0,97 và bật env.pit_fall_claw (không cộng x đi được khi đang rơi rồi chết). 819.200 bước, seed 1, GPU;
# chấm như lượt đầu (100 lượt seed 7 mỗi checkpoint, thêm 100 lượt seed 11 ở checkpoint cuối).
set -eu
cd "$(dirname "$0")/.."
DIR=runs/1-3/tuning/gamma97-claw-819k
if [ -f "$DIR.pid" ] && kill -0 "$(cat "$DIR.pid")" 2>/dev/null; then echo "lượt 1-3 E đang chạy"; exit 0; fi
if pgrep -f "[m]ario_ppo.train_joint" > /dev/null; then echo "đang có lượt huấn luyện khác; không chạy"; exit 1; fi
bash experiments/check_rules.sh > /dev/null || { echo "kiểm luật báo sai; không chạy"; exit 1; }
OVER=$("$HOME/.venvs/mario-ppo/bin/python" -c \
  "import sys; sys.path.insert(0, 'experiments'); from tuning2_report import overrides; print(overrides('c1-e0-f1-g1'))")
case "$OVER" in *gamma*|*pit_fall_claw*) echo "khóa ghi đè đã có gamma hoặc pit_fall_claw: $OVER; không chạy"; exit 1;; esac
mkdir -p runs/1-3/tuning
echo "$(date '+%F %T') khóa ghi đè: $OVER; thêm: seed=1 algo.gamma=0.97 env.pit_fall_claw=true" >> "$DIR.log"
OVER="$OVER" setsid -f nohup bash experiments/run_single_arm.sh 1-3 "$DIR" configs/group-1-3-tuning.yaml 819200 \
  "seed=1 algo.gamma=0.97 env.pit_fall_claw=true" >> "$DIR.log" 2>&1 < /dev/null
sleep 3
echo "đã bắt đầu $DIR, nhóm $(cat "$DIR.pid")"
