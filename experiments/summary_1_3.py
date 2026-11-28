"""Comparison table for 1-3 runs from their eval JSONs (docs/optimize-1-1.md, Mục 19.3-19.4).

Per checkpoint (100 episodes, seed 7): flags, mean x, Q = share of episodes whose end x is past 783 (past the jump
from the low tree onto the tall tree), share past 963 (past V3), and where episodes ended, by zone of Mục 15.1.

    python experiments/summary_1_3.py runs/1-3/tuning/first-819k runs/1-3/tuning/gamma97-claw-819k
"""
import json
import sys
from pathlib import Path

ZONES = [("<V1", 0, 257), ("V1", 257, 403), ("V1-V2", 403, 499), ("V2", 499, 660), ("V2-V3 cú nhảy", 660, 784),
         ("V3", 784, 964), ("V3-V4", 964, 1010), ("V4", 1010, 1235), ("V4-V5", 1235, 1298), ("V5", 1298, 1587),
         ("V6", 1587, 1684), ("V7", 1684, 2068), ("V8", 2068, 2426)]


def zone(x):
    for name, lo, hi in ZONES:
        if lo <= x < hi:
            return name
    return ">= cờ"


for run in sys.argv[1:]:
    run = Path(run)
    print(f"\n== {run}")
    print(f"{'step':>8} {'flag':>5} {'mean x':>7} {'Q>783':>6} {'>963':>5}  end zones (episodes without flag)")
    files = sorted(run.glob("eval/100-seed7_*.json")) + sorted(run.glob("eval/100-seed11_*.json"))
    for f in files:
        d = json.load(open(f))
        xs = d["x_pos"]
        n = len(xs)
        flags = round(d["flag_rate"] * n)
        q = sum(x > 783 for x in xs) / n
        q2 = sum(x > 963 for x in xs) / n
        zones = {}
        for x in xs:
            if x < 2425:
                zones[zone(x)] = zones.get(zone(x), 0) + 1
        tag = f.stem.split("_")[0].replace("100-", "")
        print(f"{d['global_step']:>8} {flags:>5} {sum(xs) / n:>7.0f} {q:>6.2f} {q2:>5.2f}  {tag} {zones}")
