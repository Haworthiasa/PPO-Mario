"""Bảng tổng hợp của experiments/eval_seeds.sh: mỗi checkpoint, số lượt chạm cờ /20 ở từng seed chấm, số seed qua
ngưỡng 16/20, và xác suất qua ngưỡng tính theo phân phối nhị thức từ tỉ lệ gộp. Chỉ đọc tệp.

    python experiments/eval_seeds_summary.py [--json tổng.json]
"""
import argparse
import json
import math
from pathlib import Path

OUT = Path("runs/eval-seeds")
SEEDS = [2026, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def p_pass(p, n=20, k=16):
    """P(X >= k) với X ~ Nhị thức(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = []
    print(f"{'checkpoint':<16}" + "".join(f"{s:>6}" for s in SEEDS) + f"{'gộp':>9}{'qua':>7}{'P(qua)':>8}")
    for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
        got = {}
        for s in SEEDS:
            f = d / f"20-seed{s}.json"
            if f.exists():
                r = json.loads(f.read_text(encoding="utf-8"))
                got[s] = round(r["flag_rate"] * r["episodes"])
        if not got:
            continue
        n = len(got)
        rate = sum(got.values()) / (20 * n)
        passed = sum(1 for v in got.values() if v >= 16)
        row = {"name": d.name, "flags": got, "rate": rate, "passed": passed, "seeds": n, "p_pass": p_pass(rate)}
        rows.append(row)
        print(f"{d.name:<16}" + "".join(f"{got[s]:>6}" if s in got else f"{'-':>6}" for s in SEEDS)
              + f"{100 * rate:>8.1f}%{passed:>4}/{n:<2}{p_pass(rate):>8.3f}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
