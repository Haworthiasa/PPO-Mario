"""Per-zone learning of 2-3 against the experience each zone got (docs/optimize-1-1.md, Mục 21).

For every 2-3 run and checkpoint (100 eval episodes, seed 7):
  r_z = share of episodes that reach the start of zone z (end x >= lo, or flag)
  d_z = deaths inside z / episodes that reached z
and V_z = training episodes that reached z so far, estimated as the sum over updates of
(4096 / mean episode length of that update) * r_z(step), with r_z linearly interpolated between checkpoints
(r_z = 0 at step 0 for zones past x 1000).

Question: does d_z follow V_z (experience at the zone) the same way in every run? If yes, starting some training
episodes at the zone (more V_z per step) should speed it up by the ratio of V_z rates.

    python experiments/zone_visits.py
"""
import csv
import json
from pathlib import Path

import numpy as np

RUNS = ["runs/2-3/from-scratch", "runs/2-3/from-scratch-3m", "runs/2-3/tuning/best-seed1"]
ZONES = [(1056, 1248), (1312, 1408), (1536, 1600), (1664, 2080), (2304, 2592), (2624, 3040), (3136, 3360)]
FLAG = 3593


def evals(run):
    out = []
    for f in sorted(Path(run).glob("eval/100-seed7_*.json")):
        d = json.load(open(f))
        xs = np.array(d["x_pos"])
        flag = xs >= FLAG - 1
        row = []
        for lo, hi in ZONES:
            reach = (xs >= lo) | flag
            die = (xs >= lo) & (xs < hi) & ~flag
            row.append((reach.mean(), die.sum() / max(reach.sum(), 1), int(reach.sum()), int(die.sum())))
        out.append((d["global_step"], flag.mean(), row))
    return out


def visits(run, ev):
    rows = list(csv.DictReader(open(Path(run) / "progress.csv")))
    steps = np.array([0] + [e[0] for e in ev], float)
    V = np.zeros((len(ev), len(ZONES)))
    acc = np.zeros(len(ZONES))
    prev = 0
    k = 0
    for r in rows:
        s = int(r["global_step"])
        n_ep = (s - prev) / max(float(r["ep_length"]), 1.0)
        prev = s
        for z in range(len(ZONES)):
            rz = np.array([0.0] + [e[2][z][0] for e in ev])
            acc[z] += n_ep * np.interp(s, steps, rz)
        while k < len(ev) and s >= ev[k][0]:
            V[k] = acc
            k += 1
    return V


data = {}
for run in RUNS:
    ev = evals(run)
    V = visits(run, ev)
    data[run] = (ev, V)
    print(f"\n== {run}")
    print(f"{'step':>8} {'flag':>5} " + " ".join(f"{lo:>5}-{hi:<5} d/V(k)" for lo, hi in ZONES))
    for (step, fr, row), v in zip(ev, V):
        cells = " ".join(f"{100 * d:5.1f}% {vz / 1000:5.1f}k" for (r, d, n, m), vz in zip(row, v))
        print(f"{step:>8} {fr:5.2f} {cells}")

# pool each zone over runs: deaths and reaches in bins of V_z (late checkpoints only, where reach >= 30)
print("\n== d_z by V_z bins, pooled over the three runs (deaths / reached)")
edges = [0, 500, 1000, 2000, 3000, 4500, 6000, 8000, 12000]
print(f"{'zone':>11} " + " ".join(f"{a / 1000:>4.1f}-{b / 1000:<4.1f}k" for a, b in zip(edges[:-1], edges[1:])))
for z, (lo, hi) in enumerate(ZONES):
    cells = []
    for a, b in zip(edges[:-1], edges[1:]):
        n = m = 0
        for run in RUNS:
            ev, V = data[run]
            for (step, fr, row), v in zip(ev, V):
                if a <= v[z] < b:
                    n += row[z][2]
                    m += row[z][3]
        cells.append(f"{m:>3}/{n:<5}" if n else f"{'':>9}")
    print(f"{lo:>5}-{hi:<5} " + " ".join(f"{c:>10}" for c in cells))

# visit rate per 100k steps at the end of each run, per zone
print("\n== V_z per 100k steps over the last 300k steps of each run (episodes reaching the zone)")
for run in RUNS:
    ev, V = data[run]
    i = len(ev) - 1
    j = max(0, i - 3)
    ds = (ev[i][0] - ev[j][0]) / 1e5
    print(f"{run:>28} " + " ".join(f"{(V[i][z] - V[j][z]) / ds:7.0f}" for z in range(len(ZONES))))
