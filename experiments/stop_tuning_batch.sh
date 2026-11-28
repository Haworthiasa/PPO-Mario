#!/usr/bin/env bash
# Dừng đợt chạy của experiments/run_tuning_batch.sh: SIGTERM cho cả nhóm tiến trình của script điều phối (khởi động
# bằng setsid nên số nhóm = PID của nó), chờ 10 giây, rồi SIGKILL những gì còn sống. Sau đó kiểm và báo lại.
#   bash experiments/stop_tuning_batch.sh [thư mục đợt, mặc định runs/1-1/tuning]
# Mẫu tên tiến trình bên dưới (biểu thức chính quy của pkill) khớp mọi đợt dùng configs/group-<màn>-tuning.yaml và mọi
# lần chấm checkpoint trong runs/<màn>/tuning*: các đợt 1-1 và đợt mang sang 4-1, 2-3 (run_transfer_batch.sh).
cd "$(dirname "$0")/.."
PIDFILE=${1:-runs/1-1/tuning}/batch.pid
TRAIN="configs/group-[0-9]-[0-9]-tuning.yaml"
EVAL="mario_ppo.evaluate --ckpt runs/[0-9]-[0-9]/tuning"
if [ -f "$PIDFILE" ]; then
  pg=$(cat "$PIDFILE")
  kill -TERM -- "-$pg" 2>/dev/null && echo "đã gửi SIGTERM cho nhóm $pg"
else
  echo "không thấy $PIDFILE"
fi
pkill -TERM -f "$TRAIN" 2>/dev/null
pkill -TERM -f "$EVAL" 2>/dev/null
sleep 10
if [ -f "$PIDFILE" ]; then kill -KILL -- "-$(cat "$PIDFILE")" 2>/dev/null; fi
pkill -KILL -f "$TRAIN" 2>/dev/null
pkill -KILL -f "$EVAL" 2>/dev/null
rm -f "$PIDFILE"
if pgrep -f "$TRAIN" > /dev/null || pgrep -f "$EVAL" > /dev/null; then
  echo "VẪN CÒN tiến trình:"; pgrep -af "$TRAIN"; pgrep -af "$EVAL"; exit 1
fi
echo "đã dừng hẳn"
