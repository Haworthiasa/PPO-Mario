#!/usr/bin/env bash
# Tiếp đợt 3 theo thứ tự Dũng duyệt ngày 28/11/2026 (docs/optimize-1-1.md, Mục 11): chấm nốt lượt 4-1, rồi 3 seed xác
# nhận của cấu hình 1-1 tốt nhất, rồi lượt 2-3. Dùng lại experiments/run_transfer_batch.sh với cùng các khóa ghi đè
# đã chọn ở runs/transfer/choice.log (luật chọn ở Mục 9.2). Lượt đã có checkpoint cuối thì chỉ được chấm.
#   bash experiments/start_confirm_then_2-3.sh        (trong WSL, từ thư mục gốc của kho; đợt 3 cũ phải dừng trước)
set -eu
cd "$(dirname "$0")/.."
if [ -f runs/transfer/batch.pid ] && kill -0 "$(cat runs/transfer/batch.pid)" 2>/dev/null; then
  echo "đợt 3 vẫn đang chạy (nhóm $(cat runs/transfer/batch.pid)); dừng nó trước"; exit 1
fi
OVER=$("$HOME/.venvs/mario-ppo/bin/python" experiments/tuning2_report.py --best-overrides 2>> runs/transfer/choice.log)
echo "$(date '+%F %T') tiếp đợt 3 (4-1 chấm, 1-1 seed 3-5, 2-3), khóa ghi đè: $OVER" >> runs/transfer/choice.log
OVER="$OVER" setsid -f nohup bash experiments/run_transfer_batch.sh \
  "4-1|runs/4-1/tuning/best-seed1|configs/group-4-1-tuning.yaml|614400|seed=1" \
  "1-1|runs/1-1/confirm/c1-e0-f1-g0-seed3|configs/group-1-1-tuning.yaml|614400|seed=3" \
  "1-1|runs/1-1/confirm/c1-e0-f1-g0-seed4|configs/group-1-1-tuning.yaml|614400|seed=4" \
  "1-1|runs/1-1/confirm/c1-e0-f1-g0-seed5|configs/group-1-1-tuning.yaml|614400|seed=5" \
  "2-3|runs/2-3/tuning/best-seed1|configs/group-2-3-tuning.yaml|1228800|seed=1" \
  >> runs/transfer/batch.log 2>&1 < /dev/null
sleep 3
echo "đã tiếp đợt 3, nhóm $(cat runs/transfer/batch.pid)"
