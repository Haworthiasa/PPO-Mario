#!/usr/bin/env bash
# Đợt chạy 26/11/2026 (docs/extension-directions.md, mục "Đợt chạy 26/11/2026"): ba lượt 2-1 nối tiếp checkpoint
# 6.303.744, chạy lần lượt trên CPU, mỗi lượt tự dừng ở bước 7.798.784 (stop_at_step trong cấu hình).
#
# Chạy (trong WSL, từ thư mục gốc của kho):
#   setsid nohup bash experiments/run_wall_batch.sh >> runs/2-1/batch-20261126.log 2>&1 < /dev/null &
# Dừng hẳn cả đợt (kể cả lượt đang chạy và các tiến trình môi trường của nó):
#   bash experiments/stop_wall_batch.sh
# Ctrl+C / SIGINT không dùng được cho tiến trình chạy nền; script dừng gửi SIGTERM rồi SIGKILL cho cả nhóm tiến trình.
#
# Chạy MỘT lượt khác cùng cấu hình: truyền thư mục rồi các khóa ghi đè, ví dụ lượt γ = 0,97 seed 2 (27/11/2026):
#   setsid nohup bash experiments/run_wall_batch.sh runs/2-1/gamma-0.97-cut-200-seed2 seed=2 algo.gamma=0.97 \
#       >> runs/2-1/batch-20261127.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/mario-ppo/bin/python"
CFG=configs/group-2-1-experiment.yaml
FINAL=checkpoints/step_007798784.pt
echo $$ > runs/2-1/batch.pid

run() {  # $1 = run_dir, phần còn lại = khóa ghi đè
  local dir=$1; shift
  if [ -f "$dir/$FINAL" ]; then echo "$(date '+%F %T') bỏ qua $dir (đã xong)"; return 0; fi
  if [ -e "$dir/progress.csv" ]; then
    echo "$(date '+%F %T') DỪNG: $dir có progress.csv nhưng chưa có $FINAL; xử lý tay rồi chạy lại"; exit 1
  fi
  mkdir -p "$dir"
  { git rev-parse HEAD; git status --porcelain mario_ppo configs; } > "$dir/code-version.txt"
  echo "$(date '+%F %T') bắt đầu $dir: $*"
  "$PY" -u -m mario_ppo.train_joint --config "$CFG" run_dir="$dir" "$@" > "$dir/train.log" 2>&1 &
  echo $! > "$dir/train.pid"
  wait $!
  local code=$?
  echo "$(date '+%F %T') kết thúc $dir, mã thoát $code"
  if [ $code -ne 0 ] || [ ! -f "$dir/$FINAL" ]; then echo "DỪNG ĐỢT: $dir không tới $FINAL"; exit 1; fi
}

if [ $# -gt 0 ]; then
  run "$@"
else
  run runs/2-1/wall-cut-200-seed2 seed=2
  run runs/2-1/wall-cut-2000-seed2 seed=2 env.max_no_progress_steps=2000
  run runs/2-1/gamma-0.97-cut-200 algo.gamma=0.97
fi
echo "$(date '+%F %T') xong cả đợt"
rm -f runs/2-1/batch.pid
