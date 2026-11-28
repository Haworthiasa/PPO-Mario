#!/usr/bin/env bash
# Chấm mỗi checkpoint đúng như lệnh của đề (20 lượt, evaluate.py nguyên bản) nhưng với nhiều seed chấm, để thấy điểm
# của lần chấm 20 lượt dao động thế nào. Seed 2026 là seed giảng viên dùng; lần chấm với seed đó ghi thêm video y như
# lệnh của đề. Chỉ chạy evaluate.py trên CPU, không huấn luyện, không đụng submission/.
#   setsid -f nohup bash experiments/eval_seeds.sh >> runs/eval-seeds/run.log 2>&1 < /dev/null
#   python experiments/eval_seeds_summary.py          (bảng tổng hợp)
# Tệp: runs/eval-seeds/<tên>/20-seed<S>.json. Lần chấm đã có tệp thì bỏ qua, nên chạy lại sẽ đi tiếp.
set -u
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/mario-ppo/bin/python"
OUT=runs/eval-seeds
SEEDS=(2026 1 2 3 4 5 6 7 8 9)
# tên | màn | checkpoint
CKPTS=(
  "1-1-new-seed2|1-1|runs/1-1/tuning-2/c1-e0-f1-g0/checkpoints/step_000614400.pt"
  "1-1-new-seed3|1-1|runs/1-1/confirm/c1-e0-f1-g0-seed3/checkpoints/step_000614400.pt"
  "1-1-new-seed4|1-1|runs/1-1/confirm/c1-e0-f1-g0-seed4/checkpoints/step_000614400.pt"
  "1-1-new-seed5|1-1|runs/1-1/confirm/c1-e0-f1-g0-seed5/checkpoints/step_000614400.pt"
  "1-1-submission|1-1|submission/1-1.pt"
  "4-1-new|4-1|runs/4-1/tuning/best-seed1/checkpoints/step_000614400.pt"
  "4-1-submission|4-1|submission/4-1.pt"
  "2-3-submission|2-3|submission/2-3.pt"
)
[ $# -gt 0 ] && CKPTS=("$@")
echo $$ > "$OUT/eval.pid" 2>/dev/null || { mkdir -p "$OUT"; echo $$ > "$OUT/eval.pid"; }
for spec in "${CKPTS[@]}"; do
  IFS='|' read -r name level ckpt <<< "$spec"
  if [ ! -f "$ckpt" ]; then echo "$(date '+%F %T') bỏ qua $name: chưa có $ckpt"; continue; fi
  mkdir -p "$OUT/$name"
  for s in "${SEEDS[@]}"; do
    f="$OUT/$name/20-seed$s.json"
    [ -f "$f" ] && continue
    extra=()
    [ "$s" = 2026 ] && extra=(--video "$OUT/$name/$level.mp4")
    nice -n 10 "$PY" -m mario_ppo.evaluate --ckpt "$ckpt" --level "$level" --episodes 20 --seed "$s" --out "$f" \
      "${extra[@]}" > /dev/null 2>&1 || echo "$(date '+%F %T') LỖI khi chấm $name seed $s"
  done
  echo "$(date '+%F %T') chấm xong $name"
done
echo "$(date '+%F %T') xong cả lượt chấm"
rm -f "$OUT/eval.pid"
