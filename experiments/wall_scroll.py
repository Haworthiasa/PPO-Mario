"""Chẩn đoán (không huấn luyện): ở bức tường của 2-1, xác suất bấm "trái" phụ thuộc vào phần nào của khung hình.

    python experiments/wall_scroll.py --dir <thư mục có wall-obs.npz> --ckpt <ckpt>

Đọc các quan sát do `experiments/clock_swap.py probe` ghi lại (Mario đứng ở tường) và ghi `scroll.txt` vào --dir:
  1. Dán từng vùng của khung hình giữa hai nhóm "mới tới" (đã đứng <= 200 bước) và "đứng lâu" (> 1300 bước), chỉ
     dùng các quan sát Mario ở đúng x 3026, y 79, rồi đọc xác suất "trái". Vùng nào đổi được xác suất thì mạng
     dựa vào vùng đó.
  2. Khi Mario ép vào tường, màn hình vẫn cuộn dần sang phải, nên bức tường trôi sang trái trên khung hình. Script
     tìm cột của mép trái bức tường (hàng 35 của khung 84x84) rồi lập bảng xác suất "trái" theo cột đó và theo
     thời gian đã đứng.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from mario_ppo.evaluate import load_model

LEFT, ROW = 6, 35
BINS = [(0, 200), (200, 500), (500, 850), (850, 1300), (1300, 9999)]
REGIONS = {"hàng 0-4 (trên HUD)": (slice(0, 5), slice(0, 84)), "HUD, hàng 5-10": (slice(5, 11), slice(0, 84)),
           "sân chơi, hàng 11-83": (slice(11, 84), slice(0, 84)), "sân chơi, nửa trên (11-40)": (slice(11, 41), slice(0, 84)),
           "sân chơi, nửa dưới (41-83)": (slice(41, 84), slice(0, 84))}


def wall_col(frame):
    """Cột đầu tiên của dải tối rộng >= 6 cột ở hàng ROW (mép trái bức tường); -1 nếu không thấy."""
    row = frame[ROW].astype(int)
    dark = row < np.bincount(row).argmax() - 10
    run = 0
    for c, v in enumerate(dark):
        run = run + 1 if v else 0
        if run == 6:
            return c - 5
    return -1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--ckpt", required=True)
    args = p.parse_args(argv)

    z = np.load(Path(args.dir) / "wall-obs.npz")
    obs, stood, x, y = z["obs"], z["stood"], z["x"], z["y"]
    model, ck = load_model(args.ckpt)
    model.eval()

    def p_left(o):
        with torch.no_grad():
            return np.concatenate([torch.softmax(model(torch.as_tensor(o[i:i + 512]))[0], -1)[:, LEFT].numpy()
                                   for i in range(0, len(o), 512)])

    base = p_left(obs)
    same = (x == 3026) & (y == 79)
    early, late = np.where(same & (stood <= 200))[0], np.where(same & (stood > 1300))[0]
    lines = [f"=== {args.ckpt} (bước {int(ck.get('global_step', 0))}); {len(obs)} quan sát ở tường, {same.sum()} ở x 3026, y 79 ==="]
    if len(early) and len(late):
        rng = np.random.default_rng(0)
        from_early, from_late = early[rng.integers(0, len(early), len(late))], late[rng.integers(0, len(late), len(early))]
        lines += [f"1. dán vùng giữa nhóm mới tới ({len(early)} quan sát, p(trái) {base[early].mean():.4f}) và nhóm đứng lâu "
                  f"({len(late)} quan sát, p(trái) {base[late].mean():.4f})",
                  "   vùng được dán".ljust(34) + "cảnh đứng lâu + vùng mới tới".ljust(32) + "cảnh mới tới + vùng đứng lâu"]
        for name, (r, c) in REGIONS.items():
            a, b = obs[late].copy(), obs[early].copy()
            a[:, :, r, c] = obs[from_early][:, :, r, c]
            b[:, :, r, c] = obs[from_late][:, :, r, c]
            lines.append(f"   {name}".ljust(34) + f"{p_left(a).mean():.4f}".ljust(32) + f"{p_left(b).mean():.4f}")
    col = np.array([wall_col(f) for f in obs[:, -1]])
    ok = same & (col >= 0)
    names = [f"{lo}-{hi}" if hi < 9999 else f">{lo}" for lo, hi in BINS]
    lines += ["", "2. cột của mép trái bức tường trên khung 84x84, theo thời gian đã đứng (trung bình, nhỏ nhất, lớn nhất):"]
    for n, (lo, hi) in zip(names, BINS):
        s = ok & (stood > lo) & (stood <= hi)
        if s.any():
            lines.append(f"   {n.ljust(10)} {col[s].mean():.1f}, {col[s].min()}, {col[s].max()}  ({s.sum()} quan sát)")
    lines += ["", "   p(trái) theo cột của bức tường; mỗi ô: trung bình (số quan sát), bỏ ô dưới 5 quan sát",
              "   cột".ljust(8) + "mọi lúc".ljust(16) + "".join(n.ljust(16) for n in names)]
    for c in sorted(set(col[ok].tolist())):
        sel = ok & (col == c)
        if sel.sum() < 15:
            continue
        text = f"   {c}".ljust(8) + f"{base[sel].mean():.3f} ({sel.sum()})".ljust(16)
        for lo, hi in BINS:
            s = sel & (stood > lo) & (stood <= hi)
            text += (f"{base[s].mean():.3f} ({s.sum()})" if s.sum() >= 5 else "-").ljust(16)
        lines.append(text)
    lines.append(f"   tương quan của p(trái) với cột bức tường: {np.corrcoef(col[ok], base[ok])[0, 1]:.2f}; "
                 f"với số bước đã đứng: {np.corrcoef(stood[ok], base[ok])[0, 1]:.2f}")
    (Path(args.dir) / "scroll.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
