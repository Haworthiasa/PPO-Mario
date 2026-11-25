"""Chẩn đoán (không huấn luyện, không dùng mạng): đo tọa độ x của cột cờ ở từng màn.

    python experiments/level_length.py --levels 1-1 4-1 2-1 --out runs/level-lengths/level-lengths.csv

Cách đo: cho Mario "bay" ngang qua màn bằng cách ghi thẳng vào RAM của trình giả lập ở mỗi khung hình (bất tử, giữ
độ cao cố định, giữ tốc độ chạy) và giữ nút phải + B cho tới khi chạm cột cờ; x lúc đó là vị trí cờ. Đây là công cụ
đo cho báo cáo, KHÔNG phải cách chơi và không liên quan tới mạng hay tới việc chấm: không khung hình hay phần thưởng
nào ở đây được dùng để huấn luyện. Dùng để so quãng đường giữa các màn dài ngắn khác nhau (x / x của cờ).

Kiểm chứng: nhật ký của nhóm ghi cờ 1-1 ở x = 3161, cờ 4-1 và 2-3 ở x = 3593 (đọc từ các lượt chạm cờ thật).
Cách bay chỉ hợp với màn kết thúc bằng cột cờ ngoài trời. Màn kết thúc bằng ống (dưới đất, dưới nước) hoặc bằng rìu
(lâu đài) sẽ báo "không tới cờ"; script ghi lại x xa nhất và lý do dừng thay vì đoán.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mo_gymnasium as mo

from mario_ppo.env import parse_level

RIGHT_B = 0x80 | 0x02
# Địa chỉ RAM của Super Mario Bros (bản NES)
STAR_TIMER, Y_ON_SCREEN, Y_PAGE, Y_SPEED, X_SPEED = 0x079F, 0x00CE, 0x00B5, 0x009F, 0x0057


def measure(level, height=48, max_frames=12000):
    """Trả (x cuối, chạm cờ?, số khung, lý do dừng) khi bay ngang màn ở độ cao `height` (điểm ảnh tính từ mép trên)."""
    env = mo.make("mo-supermario-v0", target=parse_level(level))
    try:
        env.reset(seed=0)
        ram = env.unwrapped.ram
        info, best_x, stalled, reason = {}, 0, 0, "frame-cap"
        for frame in range(1, max_frames + 1):
            ram[STAR_TIMER] = 30   # bất tử kiểu ngôi sao: chạm địch không chết
            ram[Y_ON_SCREEN] = height
            ram[Y_PAGE] = 1        # vẫn ở trong màn hình (0 = phía trên màn hình thì trò chơi bỏ kiểm tra va chạm)
            ram[Y_SPEED] = 0
            ram[X_SPEED] = 0x28    # tốc độ chạy tối đa
            _, _, terminated, truncated, info = env.step(RIGHT_B)
            x = int(info.get("x_pos", 0))
            if x >= 0x8000:
                x -= 0x10000
            if x > best_x:
                best_x, stalled = x, 0
            else:
                stalled += 1
            if terminated or truncated:
                reason = "flag" if info.get("flag_get") else "ended-without-flag"
                break
            if stalled >= 600:     # 10 giây không tiến: bị tường chặn hoặc màn không có cột cờ ở độ cao này
                reason = "blocked"
                break
        return best_x, bool(info.get("flag_get", False)), frame, reason
    finally:
        env.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", nargs="+", required=True)
    p.add_argument("--heights", nargs="+", type=int, default=[48, 64, 32, 96],
                   help="các độ cao thử lần lượt cho tới khi chạm cờ")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    rows = []
    for level in args.levels:
        row = None
        for height in args.heights:
            x, flag, frames, reason = measure(level, height)
            if row is None or flag or x > row["x"]:
                row = {"level": level, "flag_reached": int(flag), "x": x, "frames": frames, "height": height,
                       "stop_reason": reason}
            if flag:
                break
        rows.append(row)
        print(f"{level}: {'cờ ở x = ' if row['flag_reached'] else 'KHÔNG tới cờ, x xa nhất = '}{row['x']} "
              f"({row['frames']} khung, độ cao {row['height']}, {row['stop_reason']})", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
