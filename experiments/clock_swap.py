"""Chẩn đoán (không huấn luyện): ở bức tường của 2-1, mạng có đổi cách chơi theo đồng hồ trên màn hình không.

    python experiments/clock_swap.py probe --ckpt <ckpt> --episodes 60 --out-dir <thư mục>
    python experiments/clock_swap.py play --ckpt <ckpt> --clock freeze --episodes 100 --out-dir <thư mục>

Chỉ sửa khung hình đưa vào mạng lúc đo; không sửa trò chơi, không huấn luyện. Vùng đồng hồ trong khung 84x84 là
hàng 5..10, cột 64..83 (chữ TIME và ba chữ số; phần còn lại của HUD nằm ở cột nhỏ hơn 64).

probe  Mạng chơi như evaluate.py (cùng seed, cùng NOOP đầu lượt). Mỗi bước Mario ở tường (x 2960..3060) được đếm
       là một bước "đã đứng"; cứ --every bước lấy một quan sát. Sau đó ghép chéo: lấy cảnh (cả khung hình) của một
       nhóm theo thời gian đã đứng, dán vùng đồng hồ của một quan sát thuộc nhóm khác, rồi đọc xác suất hành động.
       Nếu mạng đọc đồng hồ thì xác suất bấm "trái" phải đi theo đồng hồ được dán, không theo cảnh.
play   Chấm hành vi. --clock none: không sửa (đối chứng). --clock freeze: từ lúc Mario tới tường, mạng luôn thấy
       đồng hồ của đúng lúc tới nơi. --clock low: từ lúc tới tường, mạng thấy đồng hồ gần hết (lấy từ một lượt đứng
       yên ở đầu màn tới khi đồng hồ còn --low-clock). Vùng đồng hồ trở lại bình thường khi Mario đã qua tường.
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

from experiments.death_look import cause_of
from mario_ppo.env import make_env
from mario_ppo.evaluate import NOOP, load_model

LEVEL = "2-1"
ROWS, COLS = slice(5, 11), slice(64, 84)
WALL_LO, WALL_HI, PASSED = 2960, 3060, 3130
BINS = [(0, 200), (200, 500), (500, 850), (850, 1300), (1300, 9999)]
ACTIONS = ["đứng yên", "phải", "phải+nhảy", "phải+chạy", "phải+chạy+nhảy", "nhảy tại chỗ", "trái"]
LEFT = 6


def probs_of(model, obs):
    out = []
    with torch.no_grad():
        for i in range(0, len(obs), 512):
            out.append(torch.softmax(model(torch.as_tensor(obs[i:i + 512]))[0], -1).numpy())
    return np.concatenate(out)


def collect(model, env_cfg, episodes, seed, every, max_noop=30):
    env = make_env(LEVEL, **dict(env_cfg, eval_mode=True))
    noops = [int(n) for n in np.random.default_rng(seed).integers(0, max_noop + 1, size=episodes)]
    torch.manual_seed(seed)
    keep = {"obs": [], "stood": [], "clock": [], "ep": [], "x": [], "y": []}
    ends = Counter()
    for i, n in enumerate(noops):
        obs, info = env.reset(seed=seed + i)
        done, steps, stood = False, 0, 0
        for _ in range(n):
            obs, _, term, trunc, info = env.step(NOOP)
            done = done or term or trunc
        while not done and steps < 3000:
            x = int(info.get("x_pos", 0))
            if WALL_LO <= x <= WALL_HI:
                stood += 1
                if stood % every == 0:
                    keep["obs"].append(np.asarray(obs).copy())
                    keep["stood"].append(stood)
                    keep["clock"].append(int(info.get("time", 0)))
                    keep["ep"].append(i)
                    keep["x"].append(x)
                    keep["y"].append(int(info.get("y_pos", 0)))
            action, _, _ = model.act(torch.as_tensor(np.asarray(obs)[None]))
            obs, _, term, trunc, info = env.step(int(action))
            done, steps = term or trunc, steps + 1
        ends[cause_of(env, info, done)] += 1
        print(f"ep {i}: {cause_of(env, info, done)}, stood {stood}", flush=True)
    env.close()
    return {k: np.asarray(v) for k, v in keep.items()}, dict(ends)


def boot(values, eps, rng, n=1000):
    """Khoảng 95% của trung bình, lấy mẫu lại theo lượt (các bước trong một lượt không độc lập)."""
    ids = np.unique(eps)
    sums = np.array([values[eps == e].sum() for e in ids])
    counts = np.array([(eps == e).sum() for e in ids])
    pick = rng.integers(0, len(ids), size=(n, len(ids)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def probe(args, model, ck, out):
    data, ends = collect(model, ck["config"]["env"], args.episodes, args.seed, args.every)
    obs, stood, clock, ep = data["obs"], data["stood"], data["clock"], data["ep"]
    np.savez_compressed(out / "wall-obs.npz", **data)
    rng = np.random.default_rng(args.seed)
    groups = [np.where((stood > lo) & (stood <= hi))[0] for lo, hi in BINS]
    names = [f"{lo}-{hi}" if hi < 9999 else f">{lo}" for lo, hi in BINS]
    lines = [f"=== {args.ckpt} (bước {int(ck.get('global_step', 0))}): {args.episodes} lượt, seed {args.seed}, "
             f"lấy 1 trên {args.every} bước ở tường ===",
             "kết thúc lượt: " + ", ".join(f"{k} {v}" for k, v in sorted(ends.items(), key=lambda kv: -kv[1]))]
    hud = obs[:, -1, ROWS, :].astype(np.int16)
    changing = np.where(hud.max(0) - hud.min(0) > 8)
    if changing[1].size:
        lines.append(f"cột của các điểm ảnh HUD có thay đổi giữa các quan sát ở tường: {changing[1].min()}.."
                     f"{changing[1].max()} (vùng được dán: cột {COLS.start}..{COLS.stop - 1})")
    lines.append("nhóm theo số bước đã đứng ở tường: " + "; ".join(
        f"{n}: {len(g)} quan sát, {len(np.unique(ep[g]))} lượt, đồng hồ trung bình "
        f"{clock[g].mean():.0f}" if len(g) else f"{n}: 0 quan sát" for n, g in zip(names, groups)))
    base = probs_of(model, obs)
    result = {"groups": names, "n": [len(g) for g in groups], "original": [], "grid": [], "ci": []}
    lines += ["", "xác suất bấm \"trái\" (trung bình); hàng = cảnh, cột = đồng hồ được dán; [khoảng 95% lấy mẫu lại theo lượt]",
              "cảnh \\ đồng hồ".ljust(16) + "không sửa".ljust(22) + "".join(n.ljust(22) for n in names)]
    for si, sg in enumerate(groups):
        if not len(sg):
            continue
        lo, hi = boot(base[sg, LEFT], ep[sg], rng)
        row = [float(base[sg, LEFT].mean())]
        cis = [[lo, hi]]
        text = names[si].ljust(16) + f"{row[0]:.3f} [{lo:.3f}-{hi:.3f}]".ljust(22)
        for cg in groups:
            if not len(cg):
                row.append(None)
                cis.append(None)
                text += "-".ljust(22)
                continue
            edited = obs[sg].copy()
            donors = cg[rng.integers(0, len(cg), size=len(sg))]
            edited[:, :, ROWS, COLS] = obs[donors][:, :, ROWS, COLS]
            p = probs_of(model, edited)[:, LEFT]
            lo, hi = boot(p, ep[sg], rng)
            row.append(float(p.mean()))
            cis.append([lo, hi])
            text += f"{p.mean():.3f} [{lo:.3f}-{hi:.3f}]".ljust(22)
        result["original"].append(row[0])
        result["grid"].append(row[1:])
        result["ci"].append(cis)
        lines.append(text)
    lines += ["", "phân phối hành động trung bình, cảnh không sửa (" + " | ".join(ACTIONS) + "):"]
    for n, g in zip(names, groups):
        if len(g):
            lines.append(f"  {n.ljust(10)} " + " | ".join(f"{v:.3f}" for v in base[g].mean(0)))
    # cùng một chỗ đứng thì xác suất "trái" có còn đổi theo thời gian đã đứng không
    xs, ground = data["x"], data["y"] == np.bincount(data["y"]).argmax()
    lines += ["", f"xác suất \"trái\" theo chỗ đứng (x) và thời gian đã đứng, cảnh không sửa; chỉ các quan sát Mario ở độ cao "
              f"hay gặp nhất (y = {np.bincount(data['y']).argmax()}); mỗi ô: trung bình (số quan sát)",
              "x".ljust(8) + "".join(n.ljust(18) for n in names) + "mọi nhóm"]
    result["by_x"] = {}
    for xv, _ in sorted(Counter(xs[ground].tolist()).most_common(12)):
        text, cells = str(xv).ljust(8), []
        for g in groups:
            sel = g[(xs[g] == xv) & ground[g]]
            cells.append([float(base[sel, LEFT].mean()), len(sel)] if len(sel) else None)
            text += (f"{base[sel, LEFT].mean():.3f} ({len(sel)})" if len(sel) else "-").ljust(18)
        sel = np.where((xs == xv) & ground)[0]
        lines.append(text + f"{base[sel, LEFT].mean():.3f} ({len(sel)})")
        result["by_x"][str(xv)] = cells
    lines += ["", "phân bố chỗ đứng (x) theo nhóm thời gian: trung bình, nhỏ nhất, lớn nhất"]
    for n, g in zip(names, groups):
        if len(g):
            lines.append(f"  {n.ljust(10)} {xs[g].mean():.1f}, {xs[g].min()}, {xs[g].max()}")
    (out / "probe.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out / "probe.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def low_clock_patch(env_cfg, target):
    """Vùng đồng hồ (4 khung) của một lượt đứng yên ở đầu màn tới khi đồng hồ còn `target`."""
    env = make_env(LEVEL, **dict(env_cfg, eval_mode=True))
    obs, info = env.reset(seed=0)
    while int(info.get("time", 400)) > target:
        obs, _, term, trunc, info = env.step(NOOP)
        if term or trunc:
            break
    env.close()
    return np.asarray(obs)[:, ROWS, COLS].copy(), int(info.get("time", 0))


def play(args, model, ck, out, max_noop=30):
    env_cfg = ck["config"]["env"]
    low, low_value = low_clock_patch(env_cfg, args.low_clock) if args.clock == "low" else (None, None)
    env = make_env(LEVEL, **dict(env_cfg, eval_mode=True))
    noops = [int(n) for n in np.random.default_rng(args.seed).integers(0, max_noop + 1, size=args.episodes)]
    torch.manual_seed(args.seed)
    rows = []
    for i, n in enumerate(noops):
        obs, info = env.reset(seed=args.seed + i)
        done, steps, stood, best, patch, left_sum, pass_stood = False, 0, 0, 0, None, 0.0, None
        for _ in range(n):
            obs, _, term, trunc, info = env.step(NOOP)
            done = done or term or trunc
        while not done and steps < 3000:
            x = int(info.get("x_pos", 0))
            best = max(best, x)
            frame = np.asarray(obs)
            if best >= WALL_LO and best <= PASSED and args.clock != "none":
                if patch is None:
                    patch = frame[:, ROWS, COLS].copy() if args.clock == "freeze" else low
                frame = frame.copy()
                frame[:, ROWS, COLS] = patch
            if WALL_LO <= x <= WALL_HI:
                stood += 1
                with torch.no_grad():
                    left_sum += float(torch.softmax(model(torch.as_tensor(frame[None]))[0], -1)[0, LEFT])
            if best > PASSED and pass_stood is None:
                pass_stood = stood
            action, _, _ = model.act(torch.as_tensor(frame[None]))
            obs, _, term, trunc, info = env.step(int(action))
            done, steps = term or trunc, steps + 1
        best = max(best, int(info.get("x_pos", 0)))
        if best > PASSED and pass_stood is None:
            pass_stood = stood
        rows.append({"episode": i, "cause": cause_of(env, info, done), "x_pos": int(info.get("x_pos", 0)),
                     "steps": steps, "reached": best >= WALL_LO, "passed": best > PASSED, "stood": stood,
                     "stood_at_pass": pass_stood, "p_left": left_sum / stood if stood else None})
        print(f"ep {i}: {rows[-1]}", flush=True)
        (out / f"play-{args.clock}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    env.close()
    reached = [r for r in rows if r["reached"]]
    passed = [r for r in rows if r["passed"]]
    total = sum(r["stood"] for r in reached)
    times = sorted(r["stood_at_pass"] for r in passed)
    lines = [f"=== {args.ckpt} (bước {int(ck.get('global_step', 0))}), đồng hồ: {args.clock}"
             + (f" (dán đồng hồ {low_value})" if low is not None else "") + f"; {len(rows)} lượt, seed {args.seed} ===",
             "kết thúc lượt: " + ", ".join(f"{k} {v}" for k, v in Counter(r["cause"] for r in rows).most_common()),
             f"lượt tới tường: {len(reached)}; lượt qua tường: {len(passed)}; tổng bước ở tường: {total}; "
             f"lượt qua trên 1000 bước ở tường: {1000 * len(passed) / max(total, 1):.2f}",
             f"số bước đã đứng ở tường lúc qua: {times}",
             f"qua sau <= 200 bước: {sum(t <= 200 for t in times)}; sau <= 850 bước: {sum(t <= 850 for t in times)}",
             "xác suất \"trái\" trung bình ở tường (theo bước): "
             f"{sum(r['p_left'] * r['stood'] for r in reached if r['stood']) / max(total, 1):.3f}"]
    (out / f"play-{args.clock}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["probe", "play"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--every", type=int, default=8, help="probe: lấy một quan sát trên chừng này bước ở tường")
    p.add_argument("--clock", choices=["none", "freeze", "low"], default="none")
    p.add_argument("--low-clock", type=int, default=40)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    torch.set_num_threads(1)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, ck = load_model(args.ckpt)
    model.eval()
    (probe if args.mode == "probe" else play)(args, model, ck, out)


if __name__ == "__main__":
    main()
