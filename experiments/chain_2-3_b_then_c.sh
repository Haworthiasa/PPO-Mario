#!/usr/bin/env bash
# Nối lượt B với lượt C của 2-3 (docs/optimize-1-1.md, Mục 23.3), để GPU không phải chờ phần chấm đủ 12 checkpoint của B:
#   1. chờ B (runs/2-3/tuning/batch2048-seed1) huấn luyện xong;
#   2. chấm ngay checkpoint cuối của B: 100 lượt seed 7 và 100 lượt seed 11, song song (run_single_arm.sh của B thấy
#      tệp đã có nên bỏ qua hai lần chấm này);
#   3. chọn nền cho C theo luật chốt trước: Y_B >= 0,70 và explained variance trung bình sau 410k bước >= 0,5 thì nền
#      là B, ngược lại là đối chứng;
#   4. bắt đầu C (experiments/start_2-3_c.sh) rồi chờ C chấm xong, để script canh máy theo dõi cả chuỗi qua tệp pid này.
#   setsid -f nohup bash experiments/chain_2-3_b_then_c.sh >> runs/2-3/tuning/chain-bc.log 2>&1 < /dev/null
set -u
cd "$(dirname "$0")/.."
PY_CPU="$HOME/.venvs/mario-ppo/bin/python"
PIDF=runs/2-3/tuning/chain-bc.pid
echo $$ > "$PIDF"
B=runs/2-3/tuning/batch2048-seed1
final=$B/checkpoints/step_001228800.pt
echo "$(date '+%F %T') chờ B huấn luyện xong"
while [ -f "$B/train.pid" ] && kill -0 "$(cat "$B/train.pid")" 2>/dev/null; do sleep 30; done
if [ ! -f "$final" ]; then echo "$(date '+%F %T') DỪNG: B không có $final"; rm -f "$PIDF"; exit 1; fi
mkdir -p "$B/eval"
for s in 7 11; do
  out="$B/eval/100-seed${s}_001228800.json"
  if [ ! -f "$out" ]; then
    "$PY_CPU" -m mario_ppo.evaluate --ckpt "$final" --level 2-3 --episodes 100 --seed "$s" --out "$out" \
      > /dev/null 2>&1 &
  fi
done
wait
read -r Y EV <<< "$("$PY_CPU" - "$B" <<'EOF'
import csv, json, sys
run = sys.argv[1]
flags = [json.load(open(f"{run}/eval/100-seed{s}_001228800.json"))["flag_rate"] for s in (7, 11)]
ev = [float(r["explained_var"]) for r in csv.DictReader(open(f"{run}/progress.csv")) if int(r["global_step"]) > 409600]
print(f"{sum(flags) / 2:.3f} {sum(ev) / len(ev):.3f}")
EOF
)"
if [ -z "${Y:-}" ]; then echo "$(date '+%F %T') DỪNG: không đọc được kết quả chấm của B"; rm -f "$PIDF"; exit 1; fi
base=$("$PY_CPU" -c "print('batch2048' if $Y >= 0.70 and $EV >= 0.5 else 'control')")
echo "$(date '+%F %T') B: Y = $Y (seed 7 + 11), explained variance sau 410k = $EV; nền cho C: $base"
bash experiments/start_2-3_c.sh "$base" || { echo "$(date '+%F %T') DỪNG: không bắt đầu được C"; rm -f "$PIDF"; exit 1; }
case "$base" in control) C=runs/2-3/tuning/restart-seed1 ;; *) C=runs/2-3/tuning/batch2048-restart-seed1 ;; esac
sleep 10
while [ -f "$C.pid" ] && kill -0 "$(cat "$C.pid")" 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') C xong: $C"
rm -f "$PIDF"
