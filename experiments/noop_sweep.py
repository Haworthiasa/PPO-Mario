"""Chẩn đoán (không huấn luyện): kết quả một màn phụ thuộc thế nào vào số bước NOOP đầu lượt.

    python experiments/noop_sweep.py --ckpt <ckpt> --level 5-2 --per-noop 10 --seed 7 --out-dir <thư mục>

`evaluate.py` bốc số bước NOOP của từng lượt (0..30) từ --seed, nên hai seed chấm hai tập NOOP khác nhau. Script
này chấm đủ 31 giá trị NOOP, mỗi giá trị --per-noop lượt lấy mẫu hành động như `evaluate.py`, và thêm một lượt
luôn chọn hành động xác suất cao nhất. Trung bình trên 31 giá trị là tỉ lệ chạm cờ không phụ thuộc vào việc seed
bốc trúng NOOP nào (`evaluate.py` bốc đều 0..30, nên kỳ vọng của nó chính là trung bình này).

Khác `evaluate.py` ở một điểm: mỗi giá trị NOOP dùng một dòng số ngẫu nhiên riêng cho việc lấy mẫu hành động
(seed + 1000 * NOOP), nên các lượt ở đây không trùng với lượt nào của `evaluate.py`.

Tệp ghi ra trong --out-dir:
    sweep.json   mỗi lượt một dòng: NOOP, kiểu chọn hành động, x cuối, số bước, lý do kết thúc
    sweep.txt    bảng theo từng giá trị NOOP và tổng
"""
import argparse
import json
import sys
from collections import Counter
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from experiments.death_look import cause_of
from mario_ppo.env import make_env
from mario_ppo.evaluate import NOOP, load_model


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def episode(model, env, noop, greedy, seed, max_steps):
    obs, info = env.reset(seed=seed)
    done, steps = False, 0
    for _ in range(noop):
        obs, _, term, trunc, info = env.step(NOOP)
        if term or trunc:
            done = True
            break
    while not done and steps < max_steps:
        x = torch.as_tensor(np.asarray(obs)[None])
        if greedy:
            with torch.no_grad():
                action = model(x)[0].argmax(-1)
        else:
            action, _, _ = model.act(x)
        obs, _, term, trunc, info = env.step(int(action))
        done, steps = term or trunc, steps + 1
    return {"noop_steps": noop, "greedy": greedy, "x_pos": int(info.get("x_pos", 0)), "steps": steps,
            "cause": cause_of(env, info, done)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--level", required=True)
    p.add_argument("--per-noop", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-noop", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    model, ck = load_model(args.ckpt)
    model.eval()
    env = make_env(args.level, **dict(ck["config"]["env"], eval_mode=True))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for n in range(args.max_noop + 1):
            rows.append(episode(model, env, n, True, args.seed, args.max_steps))
            torch.manual_seed(args.seed + 1000 * n)
            for k in range(args.per_noop):
                rows.append(episode(model, env, n, False, args.seed + k, args.max_steps))
            (out / "sweep.json").write_text(json.dumps(rows), encoding="utf-8")
            print(f"noop {n} done", flush=True)
    finally:
        env.close()

    sampled = [r for r in rows if not r["greedy"]]
    greedy = [r for r in rows if r["greedy"]]
    flags = sum(r["cause"] == "flag" for r in sampled)
    lo, hi = wilson(flags, len(sampled))
    lines = [f"=== {args.level} với {args.ckpt}: {args.max_noop + 1} giá trị NOOP x {args.per_noop} lượt lấy mẫu ===",
             f"lấy mẫu: chạm cờ {flags}/{len(sampled)} = {100 * flags / len(sampled):.1f}% "
             f"(Wilson 95% {100 * lo:.1f}-{100 * hi:.1f}, coi các lượt là độc lập)",
             f"chọn xác suất cao nhất: chạm cờ {sum(r['cause'] == 'flag' for r in greedy)}/{len(greedy)}",
             "NOOP | chạm cờ khi lấy mẫu | lượt hỏng khi lấy mẫu (lý do@x) | chọn xác suất cao nhất"]
    for n in range(args.max_noop + 1):
        mine = [r for r in sampled if r["noop_steps"] == n]
        bad = Counter(f"{r['cause']}@{r['x_pos']}" for r in mine if r["cause"] != "flag")
        g = next(r for r in greedy if r["noop_steps"] == n)
        lines.append(f"{n:4d} | {sum(r['cause'] == 'flag' for r in mine):2d}/{len(mine)} | "
                     f"{', '.join(f'{k} x{v}' if v > 1 else k for k, v in bad.most_common()) or '-'} | "
                     f"{g['cause']}@{g['x_pos']} ({g['steps']} bước)")
    (out / "sweep.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
