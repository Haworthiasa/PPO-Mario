#!/usr/bin/env bash
# Lượt 2-3 phương án B (docs/optimize-1-1.md, Mục 21.3 và 22): cấu hình c1-e0-f1-g0 như lượt đối chứng
# runs/2-3/tuning/best-seed1, chỉ đổi cách chia lô: 16 × 128 = 2.048 mẫu mỗi lần cập nhật, 8 minibatch (vẫn 256 mẫu),
# trần tốc độ học 8e-4. 1.228.800 bước, seed 1, GPU; chấm 100 lượt seed 7 mỗi checkpoint, thêm seed 11 ở checkpoint cuối.
set -eu
cd "$(dirname "$0")/.."
DIR=runs/2-3/tuning/batch2048-seed1
if [ -f "$DIR.pid" ] && kill -0 "$(cat "$DIR.pid")" 2>/dev/null; then echo "lượt 2-3 B đang chạy"; exit 0; fi
if pgrep -f "[m]ario_ppo.train_joint" > /dev/null; then echo "đang có lượt huấn luyện khác; không chạy"; exit 1; fi
bash experiments/check_rules.sh > /dev/null || { echo "kiểm luật báo sai; không chạy"; exit 1; }
BASE=$("$HOME/.venvs/mario-ppo/bin/python" -c \
  "import sys; sys.path.insert(0, 'experiments'); from tuning2_report import overrides; print(overrides('c1-e0-f1-g0'))")
case "$BASE" in *"algo.num_steps=256"*"algo.lr_control.lr_max=4e-4"*) ;; *) echo "khóa ghi đè không như mong đợi: $BASE"; exit 1;; esac
OVER=$(echo "$BASE" | sed -e 's/algo.num_steps=256/algo.num_steps=128/' -e 's/algo.lr_control.lr_max=4e-4/algo.lr_control.lr_max=8e-4/')
OVER="$OVER algo.num_minibatches=8"
mkdir -p runs/2-3/tuning
echo "$(date '+%F %T') khóa ghi đè: $OVER; thêm: seed=1" >> "$DIR.log"
OVER="$OVER" setsid -f nohup bash experiments/run_single_arm.sh 2-3 "$DIR" configs/group-2-3-tuning.yaml 1228800 \
  "seed=1" >> "$DIR.log" 2>&1 < /dev/null
sleep 3
echo "đã bắt đầu $DIR, nhóm $(cat "$DIR.pid")"
