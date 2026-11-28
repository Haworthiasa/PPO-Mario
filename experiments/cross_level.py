"""Does a zone's death rate fall with the number of training visits to it the same way on 1-1, 4-1 and 2-3?
(docs/optimize-1-1.md, Mục 22)

For every checkpoint with eval files (100 episodes, seed 7):
  d_z = deaths inside zone z / episodes that reached the start of z
  V_z = training episodes that reached z so far (as in experiments/zone_visits.py: per update,
        (4096 / mean episode length) * r_z(step), r_z interpolated between checkpoints)
Per zone, all runs of a level pooled, fit by binomial maximum likelihood
  d_z(V) = min(0.95, a_z * (V / 1000) ** -alpha_z)
Only from-scratch runs are used (a transfer run starts with visits this estimate does not see).

    python experiments/cross_level.py
"""
import csv
import json
from pathlib import Path

import numpy as np

LEVELS = {
    "1-1": dict(flag=3161, zones=[(1728, 1824), (1824, 2048), (2432, 2496), (2688, 2784)],
                runs=sorted(str(p) for p in Path("runs/1-1/tuning-2").iterdir() if (p / "progress.csv").exists())
                + sorted(str(p) for p in Path("runs/1-1/confirm").iterdir() if (p / "progress.csv").exists())),
    "4-1": dict(flag=3593, zones=[(1248, 1344), (1760, 1952), (1984, 2272), (2784, 2944)],
                runs=["runs/4-1/from-scratch", "runs/4-1/tuning/best-seed1", "runs/4-1/tuning/fast-819k"]),
    "2-3": dict(flag=3593, zones=[(1056, 1248), (1312, 1408), (1536, 1600), (1664, 2080), (2304, 2592),
                                  (2624, 3040), (3136, 3360)],
                runs=["runs/2-3/from-scratch", "runs/2-3/from-scratch-3m", "runs/2-3/tuning/best-seed1"]),
}


def evals(run, flag_x, zones):
    out = []
    for f in sorted(Path(run).glob("eval/100-seed7_*.json")):
        d = json.load(open(f))
        xs = np.array(d["x_pos"])
        flag = xs >= flag_x - 1
        row = []
        for lo, hi in zones:
            reach = (xs >= lo) | flag
            die = (xs >= lo) & (xs < hi) & ~flag
            row.append((reach.mean(), int(reach.sum()), int(die.sum())))
        out.append((d["global_step"], flag.mean(), row))
    return out


def visits(run, ev, nz):
    rows = list(csv.DictReader(open(Path(run) / "progress.csv")))
    steps = np.array([0] + [e[0] for e in ev], float)
    rz = [np.array([0.0] + [e[2][z][0] for e in ev]) for z in range(nz)]
    V = np.zeros((len(ev), nz))
    acc = np.zeros(nz)
    prev, k = 0, 0
    for r in rows:
        s = int(r["global_step"])
        n_ep = (s - prev) / max(float(r["ep_length"]), 1.0)
        prev = s
        for z in range(nz):
            acc[z] += n_ep * np.interp(s, steps, rz[z])
        while k < len(ev) and s >= ev[k][0]:
            V[k] = acc
            k += 1
    return V


def fit(points):
    """points: (V, reached, died). Binomial MLE of d = min(0.95, a (V/1000)^-alpha) on a grid."""
    V = np.array([p[0] for p in points]) / 1000
    n = np.array([p[1] for p in points], float)
    m = np.array([p[2] for p in points], float)
    best = (-np.inf, None, None)
    for alpha in np.arange(0.0, 3.001, 0.02):
        for la in np.arange(-6, 1.001, 0.02):
            d = np.clip(np.exp(la) * V ** -alpha, 1e-6, 0.95)
            ll = (m * np.log(d) + (n - m) * np.log1p(-d)).sum()
            if ll > best[0]:
                best = (ll, alpha, np.exp(la))
    ll, alpha, a = best
    # profile likelihood interval for alpha (drop of 2 in log likelihood, about 95%)
    lo_hi = []
    for alpha2 in np.arange(0.0, 3.001, 0.02):
        lls = []
        for la in np.arange(-6, 1.001, 0.02):
            d = np.clip(np.exp(la) * V ** -alpha2, 1e-6, 0.95)
            lls.append((m * np.log(d) + (n - m) * np.log1p(-d)).sum())
        if max(lls) > ll - 2:
            lo_hi.append(alpha2)
    return a, alpha, (min(lo_hi), max(lo_hi)), int(n.sum()), int(m.sum())


def run_factor(run_pts, fits):
    """One multiplier k per run on every zone's fitted curve: d = min(0.95, k a_z (V/1000)^-alpha_z).
    k < 1 means the run dies less than the pooled curve at the same number of visits."""
    best = (-np.inf, None)
    for lk in np.arange(-2.5, 2.501, 0.01):
        ll = 0.0
        for z, pts in enumerate(run_pts):
            if not pts:
                continue
            a, alpha = fits[z]
            V = np.array([p[0] for p in pts]) / 1000
            n = np.array([p[1] for p in pts], float)
            m = np.array([p[2] for p in pts], float)
            d = np.clip(np.exp(lk) * a * V ** -alpha, 1e-6, 0.95)
            ll += (m * np.log(d) + (n - m) * np.log1p(-d)).sum()
        if ll > best[0]:
            best = (ll, lk)
    # +-1 standard error from the curvature of the log likelihood
    ll0, lk0 = best
    se = []
    for lk in np.arange(lk0 - 1, lk0 + 1.0001, 0.005):
        ll = 0.0
        for z, pts in enumerate(run_pts):
            if not pts:
                continue
            a, alpha = fits[z]
            V = np.array([p[0] for p in pts]) / 1000
            n = np.array([p[1] for p in pts], float)
            m = np.array([p[2] for p in pts], float)
            d = np.clip(np.exp(lk) * a * V ** -alpha, 1e-6, 0.95)
            ll += (m * np.log(d) + (n - m) * np.log1p(-d)).sum()
        if ll > ll0 - 0.5:
            se.append(lk)
    return lk0, (max(se) - min(se)) / 2


def main():
    summary = {}
    factors = {}
    for level, spec in LEVELS.items():
        zones = spec["zones"]
        pts = [[] for _ in zones]
        per_run = {}
        ends = []
        for run in spec["runs"]:
            ev = evals(run, spec["flag"], zones)
            if not ev:
                continue
            V = visits(run, ev, len(zones))
            per_run[run] = [[] for _ in zones]
            for (step, fr, row), v in zip(ev, V):
                for z, (r, n, m) in enumerate(row):
                    if n >= 20 and v[z] >= 200:
                        pts[z].append((v[z], n, m))
                        per_run[run][z].append((v[z], n, m))
            ends.append((run, ev[-1][0], ev[-1][1], V[-1]))
        print(f"\n==== {level}: {len(ends)} runs")
        print(f"{'zone':>11} {'a (d at V=1k)':>13} {'alpha':>6} {'95% alpha':>11} {'deaths/reached':>15}")
        summary[level] = []
        fits = []
        for z, (lo, hi) in enumerate(zones):
            a, alpha, (l, h), n, m = fit(pts[z])
            fits.append((a, alpha))
            summary[level].append(dict(zone=[lo, hi], a=round(float(a), 4), alpha=round(float(alpha), 2),
                                       alpha_95=[round(float(l), 2), round(float(h), 2)], reached=n, died=m))
            print(f"{lo:>5}-{hi:<5} {100 * a:12.1f}% {alpha:6.2f} {l:5.2f}-{h:<5.2f} {m:>6}/{n:<7}")
        print("per-run factor k on the pooled curves (death at equal visits; k < 1 = learns more per visit):")
        for run in per_run:
            lk, se = run_factor(per_run[run], fits)
            factors[run] = (lk, se)
            print(f"  {run:>40} k = {np.exp(lk):5.2f}  (log k {lk:+.2f} +- {se:.2f})")
        print("visits per zone at the last checkpoint (thousands), last zone / first zone:")
        for run, step, fr, v in ends:
            print(f"  {run:>40} {step:>8} flag {fr:4.2f}  " + " ".join(f"{x / 1000:5.1f}" for x in v)
                  + f"   ratio {v[-1] / v[0]:.2f}")

    # 1-1 stage-2 design (2^(4-1) in C, E, F, G): main effects on log k
    design = {n: [int(c) for c in n[1::3]] for n in
              ["c0-e0-f0-g0", "c1-e1-f1-g1", "c1-e0-f0-g1", "c0-e1-f0-g1", "c0-e0-f1-g1", "c1-e1-f0-g0", "c1-e0-f1-g0",
               "c0-e1-f1-g0"]}
    lk = {n: factors[f"runs/1-1/tuning-2/{n}"][0] for n in design}
    print("\n1-1 design, main effect of each factor on log k (4 runs vs 4 runs), and the multiplier exp(effect):")
    for i, name in enumerate(["C lambda 0.95", "E final cap 1e-4", "F entropy to 0.001", "G start cap 6e-4"]):
        hi = np.mean([lk[n] for n in design if design[n][i] == 1])
        lo = np.mean([lk[n] for n in design if design[n][i] == 0])
        print(f"  {name:>20}: {hi - lo:+.2f}  (x{np.exp(hi - lo):.2f})")
        summary.setdefault("1-1-effects-on-log-k", {})[name] = round(float(hi - lo), 3)
    summary["run-factors"] = {r: [round(float(a), 3), round(float(b), 3)] for r, (a, b) in factors.items()}

    Path("runs/transfer").mkdir(exist_ok=True)
    json.dump(summary, open("runs/transfer/cross-level-zones.json", "w"), indent=1)
    print("\nwrote runs/transfer/cross-level-zones.json")


if __name__ == "__main__":
    main()
