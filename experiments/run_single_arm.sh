#!/usr/bin/env bash
# Huấn luyện MỘT lượt trên GPU rồi chấm nó trên CPU, giống một lượt của experiments/run_transfer_batch.sh nhưng có thư
# mục pid riêng, nên chạy được trong lúc đợt khác còn đang chấm (mỗi lúc vẫn chỉ một lượt huấn luyện).
# Chấm: 100 lượt seed 7 mỗi checkpoint, thêm 100 lượt seed 11 ở checkpoint cuối.
#   OVER="<khóa ghi đè>" setsid -f nohup bash experiments/run_single_arm.sh <màn> <thư mục lượt> <cấu hình> <số bước> \
#     "<khóa thêm>" >> <thư mục lượt>.log 2>&1 < /dev/null
# Pid của script ở <thư mục lượt>.pid (dừng hẳn: kill -TERM -- -$(cat <thư mục lượt>.pid)).
set -u
cd "$(dirname "$0")/.."
PY_GPU="$HOME/.venvs/mario-ppo-gpu/bin/python"
PY_CPU="$HOME/.venvs/mario-ppo/bin/python"
level=$1; dir=$2; cfg=$3; steps=$4; extra=${5:-}
: "${OVER:?cần biến OVER}"
final_step=$(printf '%09d' "$steps")
final="$dir/checkpoints/step_$final_step.pt"
echo $$ > "$dir.pid"
if [ -f "$final" ]; then
  echo "$(date '+%F %T') bỏ qua huấn luyện $dir (đã có $final)"
else
  if [ -e "$dir/progress.csv" ]; then echo "$(date '+%F %T') DỪNG: $dir có progress.csv nhưng chưa có $final"; rm -f "$dir.pid"; exit 1; fi
  mkdir -p "$dir"
  { git rev-parse HEAD; git status --porcelain mario_ppo configs; git diff HEAD -- mario_ppo; } > "$dir/code-version.txt"
  echo "$(date '+%F %T') bắt đầu $dir ($level): algo.total_steps=$steps $extra $OVER"
  # shellcheck disable=SC2086
  set -f
  "$PY_GPU" -u -m mario_ppo.train_joint --config "$cfg" run_dir="$dir" algo.total_steps="$steps" $extra $OVER \
    > "$dir/train.log" 2>&1 &
  set +f
  echo $! > "$dir/train.pid"
  wait $!
  code=$?
  echo "$(date '+%F %T') kết thúc $dir, mã thoát $code"
  if [ $code -ne 0 ] || [ ! -f "$final" ]; then echo "DỪNG: $dir không tới $final"; rm -f "$dir.pid"; exit 1; fi
fi
mkdir -p "$dir/eval"
for ck in "$dir"/checkpoints/step_*.pt; do
  step=$(basename "$ck" .pt); step=${step#step_}
  [ -f "$dir/eval/100-seed7_$step.json" ] || \
    "$PY_CPU" -m mario_ppo.evaluate --ckpt "$ck" --level "$level" --episodes 100 --seed 7 \
      --out "$dir/eval/100-seed7_$step.json" > /dev/null 2>&1
done
[ -f "$dir/eval/100-seed11_$final_step.json" ] || \
  "$PY_CPU" -m mario_ppo.evaluate --ckpt "$final" --level "$level" --episodes 100 --seed 11 \
    --out "$dir/eval/100-seed11_$final_step.json" > /dev/null 2>&1
echo "$(date '+%F %T') chấm xong $dir"
rm -f "$dir.pid"
