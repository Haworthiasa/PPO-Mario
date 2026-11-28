#!/usr/bin/env bash
# Đợt 2 tối ưu cấu hình ở 1-1 (docs/optimize-1-1.md, Mục 8, phương án 1): 8 lượt của thiết kế 2^(4-1) quanh cấu hình
# W (G = C·E·F) cộng một lượt đối chứng (cấu hình nộp), tất cả seed 2, mỗi lượt 614.400 bước, chạy LẦN LƯỢT trên GPU
# (venv ~/.venvs/mario-ppo-gpu). Lượt nào huấn luyện xong thì được chấm ở nền trên CPU (venv gốc) bằng evaluate.py:
# 100 lượt seed 7 cho mỗi checkpoint, thêm 100 lượt seed 11 cho checkpoint cuối. Mỗi lúc chỉ có một lượt huấn luyện và
# nhiều nhất một lần chấm chạy cùng.
#
# Chạy (trong WSL, từ thư mục gốc của kho):
#   mkdir -p runs/1-1/tuning-2 && setsid -f nohup bash experiments/run_tuning_stage2.sh >> runs/1-1/tuning-2/batch.log 2>&1 < /dev/null
# Dừng hẳn cả đợt (lượt đang chạy, lần chấm đang chạy và các tiến trình môi trường):
#   bash experiments/stop_tuning_batch.sh runs/1-1/tuning-2
# Lượt đã có checkpoint cuối thì bỏ qua, nên chạy lại lệnh trên sau khi dừng sẽ đi tiếp (lượt dở dang phải xóa tay).
set -u
cd "$(dirname "$0")/.."
PY_GPU="$HOME/.venvs/mario-ppo-gpu/bin/python"
PY_CPU="$HOME/.venvs/mario-ppo/bin/python"
CFG=configs/group-1-1-tuning.yaml
ROOT=runs/1-1/tuning-2
FINAL=checkpoints/step_000614400.pt
mkdir -p "$ROOT"
echo $$ > "$ROOT/batch.pid"

# W = lượt thắng của nửa đầu (a1-b0-c0-d1), chưa kèm trần đầu (lr_max) để mỗi lượt tự thêm, không ghi đè hai lần.
W="seed=2 algo.num_envs=16 algo.num_steps=256 algo.lr_control.kl_target=0.02 algo.lr_control.factor=1.5 algo.lr_control.lr_min=1e-5"
C1="algo.gae_lambda=0.95"                 # C: λ 0,95
E1="algo.lr_control.cap_final=1e-4"       # E: trần cuối lượt 1e-4 thay vì 0
F1="algo.ent_coef_final=0.001"            # F: entropy 0,01 giảm tuyến tính về 0,001
CONTROL="seed=2"                          # cấu hình nộp (như a0-b0-c0-d0), seed 2

score() {  # chấm mọi checkpoint của một lượt: 100 lượt seed 7; checkpoint cuối thêm 100 lượt seed 11
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

run() {  # $1 = tên lượt, phần còn lại = khóa ghi đè
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

# Thứ tự: cặp xác nhận (W ở seed 2, đối chứng ở seed 2) trước, rồi lượt có cả bốn thay đổi, rồi các lượt còn lại.
ARMS=(c0-e0-f0-g0 control c1-e1-f1-g1 c1-e0-f0-g1 c0-e1-f0-g1 c0-e0-f1-g1 c1-e1-f0-g0 c1-e0-f1-g0 c0-e1-f1-g0)
[ $# -gt 0 ] && ARMS=("$@")
for arm in "${ARMS[@]}"; do
  if [ "$arm" = control ]; then
    over="$CONTROL"
  else
    over="$W"
    if [[ $arm == *g1 ]]; then over="$over algo.lr_control.lr_max=6e-4"; else over="$over algo.lr_control.lr_max=4e-4"; fi
    [[ $arm == c1* ]] && over="$over $C1"
    [[ $arm == *e1* ]] && over="$over $E1"
    [[ $arm == *f1* ]] && over="$over $F1"
  fi
  # shellcheck disable=SC2086
  set -f; run "$arm" $over; set +f
  score "$ROOT/$arm" &
  # không để quá một lần chấm chạy cùng lúc với lượt huấn luyện kế tiếp
  while [ "$(jobs -rp | wc -l)" -gt 1 ]; do wait -n; done
done
wait
echo "$(date '+%F %T') xong cả đợt"
rm -f "$ROOT/batch.pid"
