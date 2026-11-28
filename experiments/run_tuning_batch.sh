#!/usr/bin/env bash
# Đợt chạy tối ưu cấu hình ở 1-1 (docs/optimize-1-1.md): 8 nhánh của thiết kế 2^(4-1), chạy LẦN LƯỢT trên GPU
# (venv ~/.venvs/mario-ppo-gpu), mỗi nhánh 614.400 bước. Nhánh nào huấn luyện xong thì được chấm ngay ở nền bằng
# evaluate.py trên CPU (venv gốc), trong lúc nhánh kế tiếp huấn luyện; mỗi lúc chỉ có một lượt huấn luyện.
#
# Chạy (trong WSL, từ thư mục gốc của kho):
#   setsid nohup bash experiments/run_tuning_batch.sh >> runs/1-1/tuning/batch.log 2>&1 < /dev/null &
# Dừng hẳn cả đợt (kể cả lượt đang chạy, lần chấm đang chạy và các tiến trình môi trường):
#   bash experiments/stop_tuning_batch.sh
# Nhánh đã có checkpoint cuối thì bỏ qua, nên chạy lại lệnh trên sau khi dừng sẽ đi tiếp từ nhánh dở dang (nhánh dở
# dang phải xóa tay, vì lịch tốc độ học không nối tiếp được qua một lượt bị cắt giữa chừng).
set -u
cd "$(dirname "$0")/.."
PY_GPU="$HOME/.venvs/mario-ppo-gpu/bin/python"
PY_CPU="$HOME/.venvs/mario-ppo/bin/python"
CFG=configs/group-1-1-tuning.yaml
ROOT=runs/1-1/tuning
FINAL=checkpoints/step_000614400.pt
mkdir -p "$ROOT"
echo $$ > "$ROOT/batch.pid"

# Bốn yếu tố, mỗi yếu tố hai mức; 8 lượt theo D = A·B·C (docs/optimize-1-1.md, Mục 3):
A1="algo.lr_control.kl_target=0.02 algo.lr_control.factor=1.5 algo.lr_control.lr_max=4e-4 algo.lr_control.lr_min=1e-5"
B1="algo.gamma=0.97 env.weights=[0.3,0.3,0.6,0.0,0.0]"
C1="algo.gae_lambda=0.95"
D1="algo.num_envs=16 algo.num_steps=256"

score() {  # chấm mọi checkpoint của một nhánh: 100 lượt seed 7; checkpoint cuối thêm 100 lượt seed 11
  local dir=$1
  mkdir -p "$dir/eval"
  for ck in "$dir"/checkpoints/step_*.pt; do
    local step; step=$(basename "$ck" .pt); step=${step#step_}
    [ -f "$dir/eval/100-seed7_$step.json" ] || \
      "$PY_CPU" -m mario_ppo.evaluate --ckpt "$ck" --level 1-1 --episodes 100 --seed 7 \
        --out "$dir/eval/100-seed7_$step.json" > /dev/null 2>&1
  done
  [ -f "$dir/eval/100-seed11_000614400.json" ] || \
    "$PY_CPU" -m mario_ppo.evaluate --ckpt "$dir/$FINAL" --level 1-1 --episodes 100 --seed 11 \
      --out "$dir/eval/100-seed11_000614400.json" > /dev/null 2>&1
  echo "$(date '+%F %T') chấm xong $dir"
}

run() {  # $1 = tên nhánh, phần còn lại = khóa ghi đè
  local dir=$ROOT/$1; shift
  if [ -f "$dir/$FINAL" ]; then echo "$(date '+%F %T') bỏ qua $dir (đã huấn luyện xong)"; return 0; fi
  if [ -e "$dir/progress.csv" ]; then
    echo "$(date '+%F %T') DỪNG: $dir có progress.csv nhưng chưa có $FINAL; xóa tay rồi chạy lại"; exit 1
  fi
  mkdir -p "$dir"
  { git rev-parse HEAD; git status --porcelain mario_ppo configs; git diff HEAD -- mario_ppo; } > "$dir/code-version.txt"
  echo "$(date '+%F %T') bắt đầu $dir: $*"
  "$PY_GPU" -u -m mario_ppo.train_joint --config "$CFG" run_dir="$dir" "$@" > "$dir/train.log" 2>&1 &
  echo $! > "$dir/train.pid"
  wait $!
  local code=$?
  echo "$(date '+%F %T') kết thúc $dir, mã thoát $code"
  if [ $code -ne 0 ] || [ ! -f "$dir/$FINAL" ]; then echo "DỪNG ĐỢT: $dir không tới $FINAL"; exit 1; fi
}

SCORERS=()
ARMS=(a0-b0-c0-d0 a1-b0-c0-d1 a0-b1-c0-d1 a1-b1-c0-d0 a0-b0-c1-d1 a1-b0-c1-d0 a0-b1-c1-d0 a1-b1-c1-d1)
[ $# -gt 0 ] && ARMS=("$@")   # chạy riêng vài nhánh: bash experiments/run_tuning_batch.sh a0-b0-c0-d0 ...
for arm in "${ARMS[@]}"; do
  over=""
  [[ $arm == a1* ]] && over="$over $A1"
  [[ $arm == *b1* ]] && over="$over $B1"
  [[ $arm == *c1* ]] && over="$over $C1"
  [[ $arm == *d1 ]] && over="$over $D1"
  # shellcheck disable=SC2086
  set -f; run "$arm" $over; set +f   # tắt glob để [0.3,...] không bị bash hiểu là mẫu tên tệp
  score "$ROOT/$arm" &
  SCORERS+=($!)
  # không để quá một lần chấm chạy cùng lúc với lượt huấn luyện kế tiếp
  while [ "$(jobs -rp | wc -l)" -gt 1 ]; do wait -n; done
done
wait
echo "$(date '+%F %T') xong cả đợt"
rm -f "$ROOT/batch.pid"
