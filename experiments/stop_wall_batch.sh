#!/usr/bin/env bash
# Dừng đợt chạy của experiments/run_wall_batch.sh: SIGTERM cho cả nhóm tiến trình của script điều phối (được khởi
# động bằng setsid nên số nhóm = PID của nó), chờ 10 giây, rồi SIGKILL những gì còn sống. Sau đó kiểm và báo lại.
cd "$(dirname "$0")/.."
PIDFILE=runs/2-1/batch.pid
PATTERN="mario_ppo.train_joint --config configs/group-2-1-experiment.yaml"
if [ -f "$PIDFILE" ]; then
  pg=$(cat "$PIDFILE")
  kill -TERM -- "-$pg" 2>/dev/null && echo "đã gửi SIGTERM cho nhóm $pg"
else
  echo "không thấy $PIDFILE"
fi
pkill -TERM -f "$PATTERN" 2>/dev/null
sleep 10
if [ -f "$PIDFILE" ]; then kill -KILL -- "-$(cat "$PIDFILE")" 2>/dev/null; fi
pkill -KILL -f "$PATTERN" 2>/dev/null
rm -f "$PIDFILE"
if pgrep -f "$PATTERN" > /dev/null; then echo "VẪN CÒN tiến trình huấn luyện:"; pgrep -af "$PATTERN"; exit 1; fi
echo "đã dừng hẳn"
