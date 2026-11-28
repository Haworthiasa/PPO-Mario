"""So sánh các lượt 2-3 của docs/optimize-1-1.md, Mục 21–23: đối chứng, B (2.048 mẫu mỗi lần cập nhật), C (bắt đầu
lại từ chỗ hỏng).

Với mỗi lượt có tệp chấm:
- Y (seed 7 + 11 ở checkpoint cuối) và tỉ lệ chạm cờ seed 7 theo checkpoint;
- M = tổng sqrt(approx_kl) qua các lần cập nhật;
- V_z = số lượt huấn luyện đã tới vùng z. Lượt có episodes.csv (env.failure_restart) đếm đúng từ tệp đó: một lượt
  tới vùng z nếu nó bắt đầu trước cuối vùng và kết thúc sau đầu vùng (hoặc chạm cờ). Lượt khác dùng ước lượng của
  experiments/zone_visits.py;
- k = hệ số của lượt trên đường d_z = a_z (V/1000)^-alpha_z khớp từ ba lượt 2-3 cũ (experiments/cross_level.py);
- chết / tới được theo vùng ở checkpoint cuối, gộp seed 7 và 11.

    python experiments/compare_2_3.py
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cross_level import LEVELS, evals, fit, run_factor, visits  # noqa: E402

SPEC = LEVELS["2-3"]
ZONES, FLAG = SPEC["zones"], SPEC["flag"]
RUNS = [("đối chứng", "runs/2-3/tuning/best-seed1"), ("B", "runs/2-3/tuning/batch2048-seed1"),
        ("C (nền đối chứng)", "runs/2-3/tuning/restart-seed1"),
        ("C (nền B)", "runs/2-3/tuning/batch2048-restart-seed1")]


def exact_visits(run, steps):
    rows = list(csv.DictReader(open(Path(run) / "episodes.csv")))
    acc, acc_r, out, k = np.zeros(len(ZONES)), np.zeros(len(ZONES)), [], 0
    for s in steps:
        while k < len(rows) and int(rows[k]["global_step"]) <= s:
            r = rows[k]
            x0, x1, fl, rs = int(r["x0"]), int(r["x_end"]), int(r["flag"]), int(r["restart"])
            for z, (lo, hi) in enumerate(ZONES):
                if x0 < hi and (x1 >= lo or fl):
                    acc[z] += 1
                    acc_r[z] += rs
            k += 1
        out.append((acc.copy(), acc_r.copy()))
    return out


def final_zones(run, step):
    died, reached, flags, n = np.zeros(len(ZONES)), np.zeros(len(ZONES)), 0, 0
    for s in (7, 11):
        f = Path(run) / "eval" / f"100-seed{s}_{step:09d}.json"
        if not f.exists():
            continue
        xs = np.array(json.load(open(f))["x_pos"])
        flag = xs >= FLAG - 1
        flags, n = flags + int(flag.sum()), n + len(xs)
        for z, (lo, hi) in enumerate(ZONES):
            reached[z] += ((xs >= lo) | flag).sum()
            died[z] += ((xs >= lo) & (xs < hi) & ~flag).sum()
    late_reached = reached[4]
    late_died = died[4:].sum()
    return flags / max(n, 1), n, died, reached, late_died, late_reached


def main():
    # đường tham chiếu: ba lượt cũ, như cross_level.py
    pts = [[] for _ in ZONES]
    for run in SPEC["runs"]:
        ev = evals(run, FLAG, ZONES)
        for (step, fr, row), v in zip(ev, visits(run, ev, len(ZONES))):
            for z, (r, n, m) in enumerate(row):
                if n >= 20 and v[z] >= 200:
                    pts[z].append((v[z], n, m))
    fits = [fit(p)[:2] for p in pts]
    print("đường tham chiếu (a ở V = 1k, alpha): " + ", ".join(f"{100 * a:.1f}% {al:.2f}" for a, al in fits))

    table = []
    for name, run in RUNS:
        if not (Path(run) / "progress.csv").exists() or not list(Path(run).glob("eval/100-seed7_*.json")):
            continue
        ev = evals(run, FLAG, ZONES)
        prog = list(csv.DictReader(open(Path(run) / "progress.csv")))
        M = sum(math.sqrt(max(float(r["approx_kl"]), 0.0)) for r in prog)
        exact = (Path(run) / "episodes.csv").exists()
        V_est = visits(run, ev, len(ZONES))
        V = [v for v, _ in exact_visits(run, [e[0] for e in ev])] if exact else list(V_est)
        per_run = [[] for _ in ZONES]
        for (step, fr, row), v in zip(ev, V):
            for z, (r, n, m) in enumerate(row):
                if n >= 20 and v[z] >= 200:
                    per_run[z].append((v[z], n, m))
        lk, se = run_factor(per_run, fits)
        last = ev[-1][0]
        Y, n_ep, died, reached, late_d, late_r = final_zones(run, last)
        law = np.prod([1 - min(0.95, math.exp(lk) * a * (V[-1][z] / 1000) ** -al) for z, (a, al) in enumerate(fits)])
        share = [float(r["restart_step_share"]) for r in prog if "restart_step_share" in r
                 and int(r["global_step"]) > 204800]
        print(f"\n== {name}: {run}")
        print("  chạm cờ seed 7 theo checkpoint: " + " ".join(f"{int(round(100 * e[1]))}" for e in ev)
              + f"   (checkpoint cuối {last})")
        print(f"  Y = {Y:.3f} ({n_ep} lượt), M = {M:.1f}, k = {math.exp(lk):.2f} (log k {lk:+.2f} ± {se:.2f}), "
              f"luật ở V cuối với k này: {law:.3f}")
        print("  V cuối theo vùng (nghìn): " + " ".join(f"{v / 1000:5.2f}" for v in V[-1])
              + ("   (đếm từ episodes.csv)" if exact else "   (ước lượng)"))
        if exact:
            print("  ước lượng cũ cho cùng lượt:  " + " ".join(f"{v / 1000:5.2f}" for v in V_est[-1]))
            _, Vr = exact_visits(run, [last])[0]
            print("  trong đó từ lượt bắt đầu lại: " + " ".join(f"{v / 1000:5.2f}" for v in Vr))
        if share:
            print(f"  phần bước của lượt bắt đầu lại sau 205k: trung bình {np.mean(share):.3f} "
                  f"(nhỏ nhất {min(share):.3f}, lớn nhất {max(share):.3f})")
        print("  chết / tới được ở checkpoint cuối: " + " ".join(f"{int(d)}/{int(r)}" for d, r in zip(died, reached))
              + f"   ba vùng cuối gộp {int(late_d)}/{int(late_r)} ({100 * late_d / max(late_r, 1):.1f}%)")
        table.append((name, Y, M, math.exp(lk), late_d / max(late_r, 1), died[3] / max(reached[3], 1)))

    print("\n| Lượt | Y | M | k | Ba vùng cuối chết | Đoạn cá chết |")
    print("|---|---|---|---|---|---|")
    for name, Y, M, k, late, fish in table:
        print(f"| {name} | {Y:.3f} | {M:.1f} | {k:.2f} | {100 * late:.1f}% | {100 * fish:.1f}% |")


if __name__ == "__main__":
    main()
