#!/usr/bin/env bash
# Đợt 3: mang cấu hình được chọn ở 1-1 (docs/optimize-1-1.md, Mục 9) sang 4-1 và 2-3. Mỗi lượt một màn, chạy LẦN
# LƯỢT trên GPU (venv ~/.venvs/mario-ppo-gpu); lượt nào huấn luyện xong thì được chấm ở nền trên CPU (venv gốc) bằng
# evaluate.py: 100 lượt seed 7 cho mỗi checkpoint, thêm 100 lượt seed 11 cho checkpoint cuối. Mỗi lúc chỉ có một lượt
# huấn luyện và nhiều nhất một lần chấm chạy cùng. Giữa hai lượt huấn luyện máy nghỉ REST_MIN phút (mặc định 10).
#
# Chạy (trong WSL, từ thư mục gốc của kho), OVER là các khóa ghi đè của cấu hình được chọn:
#   mkdir -p runs/transfer && OVER="..." setsid -f nohup bash experiments/run_transfer_batch.sh \
#     >> runs/transfer/batch.log 2>&1 < /dev/null
# Dừng hẳn:
#   bash experiments/stop_tuning_batch.sh runs/transfer
# Lượt đã có checkpoint cuối thì bỏ qua, nên chạy lại lệnh trên sau khi dừng sẽ đi tiếp (lượt dở dang phải xóa tay).
set -u
cd "$(dirname "$0")/.."
PY_GPU="$HOME/.venvs/mario-ppo-gpu/bin/python"
PY_CPU="$HOME/.venvs/mario-ppo/bin/python"
BATCH=runs/transfer
REST_MIN=${REST_MIN:-10}
: "${OVER:?cần biến OVER: các khóa ghi đè của cấu hình được chọn}"
mkdir -p "$BATCH"
echo $$ > "$BATCH/batch.pid"

# màn | thư mục lượt | cấu hình | số bước cuối | khóa thêm
ARMS=(
  "4-1|runs/4-1/tuning/best-seed1|configs/group-4-1-tuning.yaml|614400|seed=1"
  "2-3|runs/2-3/tuning/best-seed1|configs/group-2-3-tuning.yaml|1228800|seed=1"
)
[ $# -gt 0 ] && ARMS=("$@")

score() {  # $1 màn, $2 thư mục, $3 số bước cuối
  local level=$1 dir=$2 final; final=$(printf '%09d' "$3")
  mkdir -p "$dir/eval"
  for ck in "$dir"/checkpoints/step_*.pt; do
    local step; step=$(basename "$ck" .pt); step=${step#step_}
    [ -f "$dir/eval/100-seed7_$step.json" ] || \
      "$PY_CPU" -m mario_ppo.evaluate --ckpt "$ck" --level "$level" --episodes 100 --seed 7 \
        --out "$dir/eval/100-seed7_$step.json" > /dev/null 2>&1
  done
  [ -f "$dir/eval/100-seed11_$final.json" ] || \
    "$PY_CPU" -m mario_ppo.evaluate --ckpt "$dir/checkpoints/step_$final.pt" --level "$level" --episodes 100 \
      --seed 11 --out "$dir/eval/100-seed11_$final.json" > /dev/null 2>&1
  echo "$(date '+%F %T') chấm xong $dir"
}

first=1
for spec in "${ARMS[@]}"; do
  IFS='|' read -r level dir cfg steps extra <<< "$spec"
  final="$dir/checkpoints/step_$(printf '%09d' "$steps").pt"
  if [ -f "$final" ]; then
    echo "$(date '+%F %T') bỏ qua $dir (đã huấn luyện xong)"
  else
    if [ -e "$dir/progress.csv" ]; then
      echo "$(date '+%F %T') DỪNG: $dir có progress.csv nhưng chưa có $final; xóa tay rồi chạy lại"; exit 1
    fi
    if [ $first -eq 0 ] && [ "$REST_MIN" -gt 0 ]; then
      echo "$(date '+%F %T') nghỉ $REST_MIN phút trước lượt kế"; sleep $((REST_MIN * 60))
    fi
    mkdir -p "$dir"
    { git rev-parse HEAD; git status --porcelain mario_ppo configs; git diff HEAD -- mario_ppo; } > "$dir/code-version.txt"
    echo "$(date '+%F %T') bắt đầu $dir ($level): $extra $OVER"
    # shellcheck disable=SC2086
    set -f
    "$PY_GPU" -u -m mario_ppo.train_joint --config "$cfg" run_dir="$dir" $extra $OVER > "$dir/train.log" 2>&1 &
    set +f
    echo $! > "$dir/train.pid"
    wait $!
    code=$?
    echo "$(date '+%F %T') kết thúc $dir, mã thoát $code"
    if [ $code -ne 0 ] || [ ! -f "$final" ]; then echo "DỪNG ĐỢT: $dir không tới $final"; exit 1; fi
    first=0
  fi
  score "$level" "$dir" "$steps" &
  while [ "$(jobs -rp | wc -l)" -gt 1 ]; do wait -n; done
done
wait
echo "$(date '+%F %T') xong cả đợt"
rm -f "$BATCH/batch.pid"
