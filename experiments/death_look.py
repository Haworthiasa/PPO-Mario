"""Chẩn đoán (không huấn luyện): Mario dừng ở đâu và vì sao, theo đúng giao thức của mario_ppo.evaluate.

    python experiments/death_look.py --ckpt <ckpt> --level 5-1 --episodes 100 --seed 7 --out-dir <thư mục>

Script lặp lại các lượt của evaluate.py (cùng số bước NOOP, cùng seed, cùng cách lấy mẫu hành động), nên x cuối của
từng lượt phải trùng với tệp JSON do evaluate.py ghi. Nó thêm hai thứ evaluate.py không ghi: lý do lượt kết thúc và
ảnh màn hình ngay trước lúc kết thúc.

Lý do kết thúc đọc từ trạng thái trình giả lập sau bước cuối. Đây chỉ là số liệu chẩn đoán: mạng vẫn chỉ nhận khung
hình đã tiền xử lý, không khung hình hay con số nào ở đây được dùng để huấn luyện.
    flag   chạm cờ
    cap    hết --max-steps bước mà lượt chưa kết thúc
    time   đồng hồ của trò chơi về 0 (thường là Mario bị kẹt trước một vật cản)
    pit    rơi xuống hố (Mario ở dưới mép màn hình)
    enemy  chết khi còn trong màn hình (chạm địch, cây ăn thịt, đạn...)

Tệp ghi ra trong --out-dir:
    deaths.json            mỗi lượt một dòng: x cuối, số bước, số bước NOOP, lý do
    look.txt               bảng đếm theo lý do và theo từng khoảng x
    deaths-x<đầu khoảng>.png  ảnh các lượt chết ở khoảng x đông nhất: mỗi hàng một lượt, bốn cột là màn hình
                           61, 41, 21 và 1 khung hình trước khi lượt kết thúc
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from mario_ppo.env import make_env
from mario_ppo.evaluate import NOOP, load_model

Y_PAGE = 0x00B5              # địa chỉ RAM: trang màn hình theo chiều dọc của Mario; > 1 nghĩa là đã rơi khỏi màn hình
FRAMES_BEFORE_END = (61, 41, 21, 1)
BIN = 100                    # bề rộng một khoảng x khi gom các lượt chết


def cause_of(env, info, done):
    if info.get("flag_get", False):
        return "flag"
    if not done:
        return "cap"
    if int(info.get("time", 1)) == 0:
        return "time"
    if int(env.unwrapped.ram[Y_PAGE]) > 1:
        return "pit"
    return "enemy"


def look(ckpt, level, episodes=100, seed=7, max_steps=3000, max_noop=30, images=True):
    """Trả (danh sách kết quả từng lượt, danh sách ảnh cuối từng lượt). Vòng lặp chép từ evaluate.run_episodes.
    images=False: không giữ khung hình (ít bộ nhớ hơn nhiều khi Mario bị kẹt lâu), danh sách ảnh rỗng."""
    model, ck = load_model(ckpt)
    rng = np.random.default_rng(seed)
    noops = [int(n) for n in rng.integers(0, max_noop + 1, size=episodes)]
    env = make_env(level, **dict(ck["config"]["env"], eval_mode=True, record_frames=images))
    model.eval()
    torch.manual_seed(seed)
    rows, shots = [], []
    try:
        for i, n in enumerate(noops):
            obs, info = env.reset(seed=seed + i)
            done, steps = False, 0
            for _ in range(n):
                obs, _, term, trunc, info = env.step(NOOP)
                if term or trunc:
                    done = True
                    break
            while not done and steps < max_steps:
                action, _, _ = model.act(torch.as_tensor(np.asarray(obs)[None]))
                obs, _, term, trunc, info = env.step(int(action))
                done, steps = term or trunc, steps + 1
            if images:
                frames = env.get_wrapper_attr("frames")
                shots.append([frames[max(0, len(frames) - k)].copy() for k in FRAMES_BEFORE_END])
            rows.append({"episode": i, "x_pos": int(info.get("x_pos", 0)), "steps": steps, "noop_steps": n,
                         "cause": cause_of(env, info, done)})
    finally:
        env.close()
    return rows, shots


def summary_lines(level, ckpt, rows):
    lines = [f"=== {level} với {ckpt}: {len(rows)} lượt ==="]
    causes = Counter(r["cause"] for r in rows)
    lines.append("lý do: " + ", ".join(f"{name} {count}" for name, count in causes.most_common()))
    lines.append(f"x trung bình {np.mean([r['x_pos'] for r in rows]):.0f}")
    bins = Counter(r["x_pos"] // BIN for r in rows if r["cause"] != "flag")
    for b in sorted(bins):
        inside = Counter(r["cause"] for r in rows if r["cause"] != "flag" and r["x_pos"] // BIN == b)
        detail = ", ".join(f"{name} {count}" for name, count in inside.most_common())
        lines.append(f"  x {b * BIN:5d}-{b * BIN + BIN - 1:5d}: {bins[b]:3d}  ({detail})")
    return lines, [b for b, _ in bins.most_common(2)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--level", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--rows", type=int, default=4, help="số lượt trong mỗi ảnh")
    p.add_argument("--no-images", action="store_true", help="chỉ ghi deaths.json và look.txt, không ghi ảnh")
    args = p.parse_args(argv)

    rows, shots = look(args.ckpt, args.level, args.episodes, args.seed, args.max_steps, images=not args.no_images)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "deaths.json").write_text(json.dumps(rows, indent=1))
    lines, top_bins = summary_lines(args.level, args.ckpt, rows)

    import imageio.v3 as iio
    for b in ([] if args.no_images else top_bins):
        chosen = sorted((r for r in rows if r["cause"] != "flag" and r["x_pos"] // BIN == b),
                        key=lambda r: r["x_pos"])[:args.rows]
        sheet = np.concatenate([np.concatenate(shots[r["episode"]], axis=1) for r in chosen], axis=0)
        name = f"deaths-x{b * BIN}.png"
        iio.imwrite(out / name, sheet)
        lines.append(f"{name}: các hàng từ trên xuống là lượt " +
                     "; ".join(f"{r['episode']} (x {r['x_pos']}, {r['cause']})" for r in chosen))
    (out / "look.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
