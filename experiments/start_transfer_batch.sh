#!/usr/bin/env bash
# Bắt đầu đợt 3 (experiments/run_transfer_batch.sh) với cấu hình chọn theo luật ghi trước ở Mục 9.2 của
# docs/optimize-1-1.md: lượt có Y cao nhất trong 8 lượt thiết kế của đợt 2. Nếu đợt 2 chưa chấm đủ 8 lượt thì không
# chạy gì. Lựa chọn và các khóa ghi đè được ghi vào runs/transfer/choice.log.
#   bash experiments/start_transfer_batch.sh        (trong WSL, từ thư mục gốc của kho)
set -eu
cd "$(dirname "$0")/.."
mkdir -p runs/transfer
if [ -f runs/transfer/batch.pid ] && kill -0 "$(cat runs/transfer/batch.pid)" 2>/dev/null; then
  echo "đợt 3 đang chạy (nhóm $(cat runs/transfer/batch.pid))"; exit 0
fi
OVER=$("$HOME/.venvs/mario-ppo/bin/python" experiments/tuning2_report.py --best-overrides 2>> runs/transfer/choice.log)
echo "$(date '+%F %T') khóa ghi đè: $OVER" >> runs/transfer/choice.log
OVER="$OVER" setsid -f nohup bash experiments/run_transfer_batch.sh >> runs/transfer/batch.log 2>&1 < /dev/null
sleep 3
echo "đã bắt đầu đợt 3, nhóm $(cat runs/transfer/batch.pid)"
