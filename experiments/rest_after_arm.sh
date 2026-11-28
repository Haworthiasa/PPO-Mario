#!/usr/bin/env bash
# Cho máy nghỉ giữa hai lượt của một đợt chạy (experiments/run_tuning_stage2.sh hoặc run_tuning_batch.sh).
# Cách làm: chờ dòng "kết thúc <thư mục đợt>/<lượt>," trong batch.log, rồi tạm dừng (SIGSTOP) cả nhóm tiến trình của
# đợt: script điều phối, lượt kế tiếp vừa bắt đầu và lần chấm đang chạy. Sau <phút> phút thì cho chạy tiếp (SIGCONT).
# Không mất gì, tiến trình chỉ đứng yên. Một chỗ lệch duy nhất: sps của lượt kế tiếp được tính trên thời gian thực từ
# lúc lượt bắt đầu, nên con số sps ghi lại của lượt đó thấp hơn thực tế (thời gian nghỉ bị tính vào).
# Nếu script canh máy (watchdog) đã báo TRIGGER trong lúc nghỉ thì KHÔNG cho chạy tiếp, để giữ lần tạm dừng an toàn.
#
# Chạy (trong WSL, từ thư mục gốc của kho):
#   setsid -f nohup bash experiments/rest_after_arm.sh runs/1-1/tuning-2 <lượt> 15 [log của watchdog] \
#     >> runs/1-1/tuning-2/rest.log 2>&1 < /dev/null
# Bỏ nghỉ, cho chạy tiếp ngay:
#   kill -CONT -- -$(cat runs/1-1/tuning-2/batch.pid)
set -u
cd "$(dirname "$0")/.."
ROOT=$1
ARM=$2
MIN=${3:-15}
WDLOG=${4:-}
LOG=$ROOT/batch.log
pg=$(cat "$ROOT/batch.pid")
echo "$(date '+%F %T') chờ $ROOT/$ARM huấn luyện xong, rồi cho nhóm $pg nghỉ $MIN phút"
until grep -q "kết thúc $ROOT/$ARM," "$LOG"; do
  if ! kill -0 "$pg" 2>/dev/null; then echo "$(date '+%F %T') đợt không còn chạy, bỏ"; exit 1; fi
  sleep 5
done
kill -STOP -- "-$pg"
echo "$(date '+%F %T') đã tạm dừng nhóm $pg; nghỉ $MIN phút"
sleep $((MIN * 60))
if [ -n "$WDLOG" ] && grep -q TRIGGER "$WDLOG"; then
  echo "$(date '+%F %T') watchdog đã báo TRIGGER trong lúc nghỉ; để nguyên trạng thái tạm dừng"
  exit 2
fi
kill -CONT -- "-$pg"
echo "$(date '+%F %T') cho nhóm $pg chạy tiếp"
