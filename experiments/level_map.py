"""Chẩn đoán (không huấn luyện, không dùng mạng): vẽ bản đồ cả màn và chồng lên đó chỗ Mario chết khi chấm.

    python experiments/level_map.py --level 2-3 --runs runs/2-3/from-scratch runs/2-3/from-scratch-3m \
        --out docs/reports/figures/2-3-map.png [--json runs/level-maps/2-3.json]

1. Bản đồ: cho Mario "bay" ngang màn như experiments/level_length.py (ghi RAM: bất tử, giữ độ cao, giữ tốc độ chạy,
   giữ phải + B). Mỗi khung hình, mép trái màn hình ở tọa độ x của màn = trang (RAM 0x071A) · 256 + RAM 0x071C. Script
   dán 16 cột ngoài cùng bên phải của khung hình vào đúng chỗ đó trên một tấm ảnh dài, nên Mario (ở giữa màn hình)
   không lọt vào ảnh. Đây là công cụ vẽ cho báo cáo: không khung hình nào ở đây được dùng để huấn luyện.
2. Chỗ chết: đọc mọi tệp 100-seed*_*.json do evaluate.py ghi trong <run>/eval. Mỗi lượt không chạm cờ dừng ở một x
   cuối. Với mỗi khoảng 32 điểm ảnh: số lượt chết trong khoảng / số lượt đi tới đầu khoảng = xác suất chết có điều kiện.
   Các checkpoint được chia ba giai đoạn theo tỉ lệ chạm cờ của chính checkpoint (< 30%, 30–80%, >= 80%).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

RIGHT_B = 0x80 | 0x02
STAR_TIMER, Y_ON_SCREEN, Y_PAGE, Y_SPEED, X_SPEED = 0x079F, 0x00CE, 0x00B5, 0x009F, 0x0057
SCREEN_PAGE, SCREEN_X = 0x071A, 0x071C
BIN = 32


def fly_map(level, height=48, max_frames=12000, strip=16):
    """Ảnh RGB cả màn (240 x độ dài) và x của cờ."""
    import mo_gymnasium as mo
    from mario_ppo.env import parse_level

    env = mo.make("mo-supermario-v0", target=parse_level(level))
    try:
        env.reset(seed=0)
        base = env.unwrapped
        ram = base.ram
        canvas = np.zeros((240, 4600, 3), dtype=np.uint8)
        right, best_x, stalled, info = 0, 0, 0, {}
        for _ in range(max_frames):
            ram[STAR_TIMER] = 30
            ram[Y_ON_SCREEN] = height
            ram[Y_PAGE] = 1
            ram[Y_SPEED] = 0
            ram[X_SPEED] = 0x28
            _, _, terminated, truncated, info = env.step(RIGHT_B)
            left = int(ram[SCREEN_PAGE]) * 256 + int(ram[SCREEN_X])
            frame = np.asarray(base.screen)
            lo = left + 256 - strip
            if left + 256 <= canvas.shape[1]:
                if right == 0:
                    canvas[:, left:left + 256] = frame
                else:
                    canvas[:, lo:left + 256] = frame[:, 256 - strip:]
                right = max(right, left + 256)
            x = int(info.get("x_pos", 0))
            best_x, stalled = (x, 0) if x > best_x else (best_x, stalled + 1)
            if terminated or truncated or stalled >= 600:
                break
        return canvas[:, :right], best_x, bool(info.get("flag_get", False))
    finally:
        env.close()


def deaths(runs):
    """[(flag_rate của checkpoint, [x cuối của các lượt không chạm cờ], số lượt, x cờ)] từ các tệp chấm."""
    out = []
    for run in runs:
        for f in sorted(glob.glob(str(Path(run) / "eval" / "100-seed*_*.json"))):
            d = json.load(open(f, encoding="utf-8"))
            xs = d["x_pos"]
            flag_x = max(xs)
            out.append((d["flag_rate"], xs, len(xs), d.get("steps")))
    return out


def hazard(records, flag_x, lo_rate, hi_rate):
    """Xác suất chết có điều kiện theo khoảng BIN điểm ảnh, gộp các checkpoint có flag_rate trong [lo, hi)."""
    nb = flag_x // BIN + 1
    died = np.zeros(nb)
    reached = np.zeros(nb)
    n_ck = 0
    for rate, xs, _, _ in records:
        if not (lo_rate <= rate < hi_rate):
            continue
        n_ck += 1
        for x in xs:
            b = min(max(x, 0) // BIN, nb - 1)
            reached[: b + 1] += 1
            if x < flag_x:
                died[b] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(reached > 0, died / reached, np.nan)
    return h, died, reached, n_ck


def zones(died, reached, min_deaths):
    """Gộp các khoảng liền nhau có chết thành vùng; trả các vùng có >= min_deaths lượt chết."""
    out, cur = [], None
    for b, d in enumerate(died):
        if d > 0:
            if cur and b - cur["end"] <= 1:
                cur["end"] = b
                cur["died"] += d
            else:
                if cur:
                    out.append(cur)
                cur = {"start": b, "end": b, "died": d}
    if cur:
        out.append(cur)
    res = []
    for z in out:
        if z["died"] >= min_deaths:
            z["reached"] = reached[z["start"]]
            z["rate"] = z["died"] / z["reached"] if z["reached"] else float("nan")
            z["x0"], z["x1"] = z["start"] * BIN, (z["end"] + 1) * BIN
            res.append(z)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--json", default=None)
    p.add_argument("--height", type=int, default=48)
    a = p.parse_args(argv)

    img, flag_x_fly, flag_ok = fly_map(a.level, a.height)
    recs = deaths(a.runs)
    flag_x = max(max(xs) for _, xs, _, _ in recs)
    steps_flag = [s for _, xs, _, st in recs if st for x, s in zip(xs, st) if x >= flag_x]
    stages = [("< 30%", 0.0, 0.3), ("30–80%", 0.3, 0.8), ("≥ 80%", 0.8, 1.01)]
    summary = {"level": a.level, "flag_x": flag_x, "flag_x_fly": flag_x_fly, "fly_reached_flag": flag_ok,
               "checkpoints": len(recs), "episodes": sum(r[2] for r in recs),
               "steps_to_flag_median": float(np.median(steps_flag)) if steps_flag else None,
               "px_per_step": flag_x / float(np.median(steps_flag)) if steps_flag else None, "stages": {}}
    print(f"{a.level}: cờ ở x = {flag_x} (bay: {flag_x_fly}, {'tới cờ' if flag_ok else 'không tới cờ'}); "
          f"{len(recs)} lần chấm, {summary['episodes']} lượt")
    if steps_flag:
        print(f"  số bước tác tử tới cờ (trung vị) {summary['steps_to_flag_median']:.0f}, "
              f"{summary['px_per_step']:.1f} điểm ảnh mỗi bước")
    for name, lo, hi in stages:
        h, died, reached, n_ck = hazard(recs, flag_x, lo, hi)
        zs = zones(died, reached, min_deaths=max(3, int(0.01 * reached[0])) if reached[0] else 3)
        summary["stages"][name] = {"checkpoints": n_ck, "episodes": int(reached[0]),
                                   "zones": [{k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
                                              for k, v in z.items()} for z in zs]}
        print(f"  giai đoạn {name}: {n_ck} lần chấm, {int(reached[0])} lượt")
        for z in sorted(zs, key=lambda z: -z["died"])[:8]:
            print(f"    x {z['x0']:>4}–{z['x1']:<4} chết {int(z['died']):>4} / tới {int(z['reached']):>5}  "
                  f"= {100 * z['rate']:5.1f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    half = (flag_x + 64) // 2
    fig = plt.figure(figsize=(14, 7.2), dpi=130)
    fig.text(0.01, 0.985, f"Màn {a.level}: bản đồ (bay ngang bằng RAM, chỉ để vẽ) và xác suất chết có điều kiện khi chấm "
             f"theo từng khoảng {BIN} điểm ảnh", va="top", fontsize=10.5, color="#2b2a27")
    colors = {"< 30%": "#a3a198", "30–80%": "#eda100", "≥ 80%": "#e34948"}
    for row in range(2):
        x0, x1 = row * half, min((row + 1) * half, img.shape[1])
        top = 0.53 - row * 0.49
        ax_img = fig.add_axes([0.05, top + 0.22, 0.93, 0.21])
        ax_img.imshow(img[16:232, x0:x1], aspect="auto", extent=(x0, x1, 0, 1), interpolation="nearest")
        ax_img.set_yticks([])
        ax_img.set_xticks([])
        ax_h = fig.add_axes([0.05, top + 0.02, 0.93, 0.19], sharex=ax_img)
        for name, lo, hi in stages:
            h, died, reached, n_ck = hazard(recs, flag_x, lo, hi)
            if n_ck == 0:
                continue
            xs = np.arange(len(h)) * BIN
            ax_h.step(xs, 100 * np.nan_to_num(h), where="post", color=colors[name], linewidth=1.3,
                      label=f"checkpoint chạm cờ {name} ({n_ck} lần chấm)")
        ax_h.set_xlim(x0, x1)
        ax_h.set_ylim(0, 40)
        ax_h.set_yticks([0, 10, 20, 30, 40])
        ax_h.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax_h.grid(axis="y", color="#e1e0d9", linewidth=0.7)
        ax_h.tick_params(colors="#898781", length=0)
        ax_h.spines["left"].set_visible(False)
        ax_h.set_xlabel("x trong màn (điểm ảnh)", color="#52514e")
        if row == 0:
            ax_h.legend(loc="upper left", frameon=False, fontsize=8, ncol=3)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("đã ghi", a.out)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
