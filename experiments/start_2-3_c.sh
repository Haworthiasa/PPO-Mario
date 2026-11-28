#!/usr/bin/env bash
# Lượt 2-3 phương án C (docs/optimize-1-1.md, Mục 21.3 và 23): bật env.failure_restart (một phần lượt bắt đầu lại ngay
# trước chỗ vừa chết) trên một cấu hình nền đã chạy, không đổi gì khác. 1.228.800 bước, seed 1, GPU; chấm 100 lượt
# seed 7 mỗi checkpoint, thêm seed 11 ở checkpoint cuối.
#   bash experiments/start_2-3_c.sh control     # nền: c1-e0-f1-g0 như runs/2-3/tuning/best-seed1
#   bash experiments/start_2-3_c.sh batch2048   # nền: như runs/2-3/tuning/batch2048-seed1 (phương án B)
#   bash experiments/start_2-3_c.sh batch2048 1843200   # cùng cấu hình, số bước khác (Mục 24); thư mục thêm hậu tố
#     số bước, ví dụ runs/2-3/tuning/batch2048-restart-1.84m-seed1
set -eu
cd "$(dirname "$0")/.."
base=${1:?cần control hoặc batch2048}
STEPS=${2:-1228800}
case "$STEPS" in *[!0-9]*|"") echo "số bước không hợp lệ: $STEPS"; exit 1 ;; esac
if [ $((STEPS % 102400)) -ne 0 ]; then echo "số bước phải chia hết cho 102400 (chu kỳ checkpoint): $STEPS"; exit 1; fi
case "$base" in
  control) DIR=runs/2-3/tuning/restart ;;
  batch2048) DIR=runs/2-3/tuning/batch2048-restart ;;
  *) echo "nền không hợp lệ: $base"; exit 1 ;;
esac
if [ "$STEPS" -ne 1228800 ]; then
  DIR="$DIR-$(awk -v s="$STEPS" 'BEGIN { printf "%.2fm", s / 1e6 }')"
fi
DIR="$DIR-seed1"
if [ -f "$DIR.pid" ] && kill -0 "$(cat "$DIR.pid")" 2>/dev/null; then echo "lượt $DIR đang chạy"; exit 0; fi
if pgrep -f "[m]ario_ppo.train_joint" > /dev/null; then echo "đang có lượt huấn luyện khác; không chạy"; exit 1; fi
bash experiments/check_rules.sh > /dev/null || { echo "kiểm luật báo sai; không chạy"; exit 1; }
OVER=$("$HOME/.venvs/mario-ppo/bin/python" -c \
  "import sys; sys.path.insert(0, 'experiments'); from tuning2_report import overrides; print(overrides('c1-e0-f1-g0'))")
case "$OVER" in *"algo.num_steps=256"*"algo.lr_control.lr_max=4e-4"*) ;; *) echo "khóa ghi đè không như mong đợi: $OVER"; exit 1;; esac
if [ "$base" = batch2048 ]; then
  OVER=$(echo "$OVER" | sed -e 's/algo.num_steps=256/algo.num_steps=128/' -e 's/algo.lr_control.lr_max=4e-4/algo.lr_control.lr_max=8e-4/')
  OVER="$OVER algo.num_minibatches=8"
fi
RESTART="env.failure_restart.prob=0.5 env.failure_restart.pool=4 env.failure_restart.max_uses=4"
RESTART="$RESTART env.failure_restart.backup_every=16 env.failure_restart.min_age=6 env.failure_restart.max_age=48"
RESTART="$RESTART env.failure_restart.min_time=100 env.failure_restart.spares=3"
mkdir -p runs/2-3/tuning
echo "$(date '+%F %T') nền $base, $STEPS bước; khóa ghi đè: $OVER; thêm: seed=1 $RESTART" >> "$DIR.log"
OVER="$OVER" setsid -f nohup bash experiments/run_single_arm.sh 2-3 "$DIR" configs/group-2-3-tuning.yaml "$STEPS" \
  "seed=1 $RESTART" >> "$DIR.log" 2>&1 < /dev/null
sleep 3
echo "đã bắt đầu $DIR, nhóm $(cat "$DIR.pid")"
