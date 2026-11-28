"""Diagnosis only (no training): why the 1-3 policy jumps into the pit at x 640 (docs/optimize-1-1.md, Mục 18).

    python experiments/jump_returns.py <ckpt> [N=200] [SEARCH=2000] [out.json]

At a fixed state s0 (Mario on the low tree of 1-3, reached by deterministic replay of a sampled prefix in the
training env), compare the discounted return of
  P    the trained policy from s0
  S+P  a scripted jump that lands on the tall tree (search includes braking with LEFT), then the policy
  S+U  the same jump, then uniform random actions without LEFT
  S+B  the same jump, then the best of SEARCH random persistent continuations (a competent continuation)
under three reward rules and several gamma:
  R0     the current training reward: (dx + dtime)/30, death -50/30
  Rclaw  R0, but on death the x gained since the last step on the ground is taken back (fall drift not credited)
  Rland  R0, but x gained in the air is paid only when Mario lands; forfeited if he dies first
Also: the upper bound for the jump under R0 (after landing, run at full speed forever and never die), and the
gamma at which it breaks even with P.
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/mnt/d/mario-ppo")
from mario_ppo.env import make_env
from mario_ppo.evaluate import load_model
from mario_ppo.wrappers import FrameSkip

CK = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SEARCH = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
OUT = sys.argv[4] if len(sys.argv) > 4 else None
torch.set_num_threads(4)
GAMMAS = [0.9, 0.95, 0.97, 0.98, 0.99]
RULES = ["R0", "Rclaw", "Rland"]
FLOAT, Y_SCR, Y_PAGE = 0x001D, 0x00CE, 0x00B5
DEATH = 2.0 * (-25.0 * 5 / 150)

model, ck = load_model(CK)
model.eval()
algo = ck["config"]["algo"]
G_TR, L_TR = float(algo["gamma"]), float(algo["gae_lambda"])
cfg = dict(ck["config"]["env"])
cfg.update(eval_mode=False, noop_max=0, normalize_reward=False)


def build():
    e = make_env("1-3", **cfg)
    f = e
    while not isinstance(f, FrameSkip):
        f = f.env
    return e, e.unwrapped, f


env, base, fs = build()
env2, base2, fs2 = build()  # separate emulator for the head search (its _backup must not touch env's reset point)
print(f"ckpt {CK}  train gamma {G_TR} lambda {L_TR}")


def st(b, info):
    return int(info.get("x_pos", 0)), int(b.ram[Y_SCR]), int(b.ram[FLOAT]), int(b.ram[Y_PAGE])


def on_tall(x, y, fl, yp):
    return 662 <= x <= 740 and fl == 0 and yp == 1 and y < 80


def policy(obs):
    with torch.no_grad():
        logits, value = model(torch.as_tensor(np.asarray(obs)[None]))
    p = torch.softmax(logits, -1)[0].numpy().astype(np.float64)
    return p / p.sum(), float(value.reshape(-1)[0])


def find_prefix():
    for s in range(100):
        rng = np.random.default_rng(s)
        obs, info = env.reset(seed=0)
        acts = []
        for _ in range(300):
            x, y, fl, yp = st(base, info)
            if acts and 590 <= x <= 625 and fl == 0 and yp == 1:
                return acts, (x, y)
            p, _ = policy(obs)
            a = int(rng.choice(7, p=p))
            obs, _, term, trunc, info = env.step(a)
            acts.append(a)
            if term or trunc:
                break
    raise SystemExit("no prefix")


PREFIX, S0 = find_prefix()


def goto_s0(e=env, b=base):
    obs, info = e.reset(seed=0)
    for a in PREFIX:
        obs, _, _, _, info = e.step(a)
    assert st(b, info)[:2] == S0, (st(b, info), S0)
    return obs, info


_, info0 = goto_s0()
print(f"prefix {len(PREFIX)} steps, s0 at x {S0[0]} y {S0[1]}, time {info0['time']}")

# ---- head search on env2 with snapshots
goto_s0(env2, base2)
base2._backup()
HEADS = []
for brake in range(0, 5):
    for creep in range(0, 4):
        for stop in range(0, 3):
            for up in range(1, 4):
                for drift in range(5, 12):
                    seq = [6] * brake + [1] * creep + [0] * stop + [5] * up + [2] * drift + [1] * 15
                    base2._restore()
                    base2.done = False
                    for i, a in enumerate(seq):
                        _, _, term, trunc, info = fs2.step(a)
                        s = st(base2, info)
                        if on_tall(*s):
                            HEADS.append((seq[:i + 1], (brake, creep, stop, up, drift)))
                            break
                        if term or trunc or s[3] > 1:
                            break
print(f"landing jumps found: {len(HEADS)} of {5 * 4 * 3 * 3 * 7}; by brake steps "
      f"{np.bincount([h[1][0] for h in HEADS], minlength=5).tolist()}; shortest {min(len(h[0]) for h in HEADS) if HEADS else None} steps")


def rollout(tail, rng, head=None, max_steps=400):
    obs, info = goto_s0()
    rs, vs, xs, fls = [], [], [], []
    landed = None
    for t in range(max_steps):
        p, v = policy(obs)
        vs.append(v)
        if head is not None and t < len(head):
            a = head[t]
        elif tail == "policy":
            a = int(rng.choice(7, p=p))
        else:
            a = int(rng.integers(0, 6))
        obs, r, term, trunc, info = env.step(a)
        s = st(base, info)
        rs.append(float(r))
        xs.append(s[0])
        fls.append(s[2])
        if landed is None and on_tall(*s):
            landed = t + 1
        if term or trunc:
            flag = bool(info.get("flag_get", False))
            end = "flag" if flag else ("death" if term else "trunc")
            if head is not None and landed is None:
                end = "nolanding"
            return dict(r=rs, v=vs, x=xs, fl=fls, end=end, landed=landed, v_last=0.0 if term else policy(obs)[1])
    return dict(r=rs, v=vs, x=xs, fl=fls, end="cap", landed=landed, v_last=policy(obs)[1])


def variants(r0, xs, fls, died, x0):
    out = {"R0": list(r0)}
    claw, xg = list(r0), x0
    for t in range(len(r0)):
        if fls[t] == 0 and not (died and t == len(r0) - 1):
            xg = xs[t]
    if died:
        claw[-1] -= max(0, xs[-1] - xg) / 30.0
    out["Rclaw"] = claw
    land, pend, xp = [], 0.0, x0
    for t in range(len(r0)):
        dx = (xs[t] - xp) / 30.0
        xp = xs[t]
        rest = r0[t] - dx
        if fls[t] == 0 and not (died and t == len(r0) - 1):
            land.append(rest + pend + dx)
            pend = 0.0
        else:
            pend += dx
            land.append(rest)
    out["Rland"] = land
    return out


def disc(rs, g):
    return float(sum(r * g ** t for t, r in enumerate(rs)))


def gae0(ro):
    v = ro["v"] + [ro["v_last"]]
    adv = 0.0
    for t in reversed(range(len(ro["r"]))):
        adv = ro["r"][t] + G_TR * v[t + 1] - v[t] + G_TR * L_TR * adv
    return adv


def summarise(name, ros):
    ros = [r for r in ros if r["end"] != "nolanding"]
    if not ros:
        print(f"\n== {name}: no landed rollouts")
        return None
    G = {}
    for rule in RULES:
        for g in GAMMAS:
            vals = [disc(variants(r["r"], r["x"], r["fl"], r["end"] == "death", S0[0])[rule], g) for r in ros]
            G[f"{rule}|{g}"] = (float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals))))
    X = [r["x"][-1] for r in ros]
    s = dict(name=name, n=len(ros), steps=float(np.mean([len(r["r"]) for r in ros])),
             ends={e: sum(r["end"] == e for r in ros) for e in ["death", "trunc", "cap", "flag"]},
             x_end_q=[float(q) for q in np.percentile(X, [10, 50, 90])], G=G,
             gae0=float(np.mean([gae0(r) for r in ros])), v_s0=ros[0]["v"][0],
             death_x_hist=np.histogram([r["x"][-1] for r in ros if r["end"] == "death"],
                                       bins=[0, 700, 740, 800, 850, 900, 1000, 1300, 2500])[0].tolist())
    print(f"\n== {name}: n {s['n']} ends {s['ends']} steps {s['steps']:.1f} x_end q10/50/90 {s['x_end_q']}")
    print(f"   death x hist (<700,700-740,740-800,800-850,850-900,900-1000,1000-1300,>1300): {s['death_x_hist']}")
    for rule in RULES:
        print(f"   {rule:5s}: " + "  ".join(f"g{g}={G[f'{rule}|{g}'][0]:6.2f}" for g in GAMMAS))
    print(f"   GAE at t0 under R0 (critic, g {G_TR}, lam {L_TR}): {s['gae0']:.3f}   V(s0) {s['v_s0']:.3f}")
    return s


t0 = time.time()
res = dict(ckpt=CK, prefix_len=len(PREFIX), s0=S0, n_heads=len(HEADS),
           heads_by_brake=np.bincount([h[1][0] for h in HEADS], minlength=5).tolist() if HEADS else [])
rng = np.random.default_rng(1)
P = [rollout("policy", rng) for _ in range(max(20, N // 4))]
res["P"] = summarise("P: policy from s0", P)
print("   P x by step:", P[0]["x"], " fl by step:", P[0]["fl"])
print(f"   ({time.time() - t0:.0f}s)")
if not HEADS:
    print("no landing jump found from this s0")
    if OUT:
        json.dump(res, open(OUT, "w"), indent=1)
    sys.exit(0)

SP = [rollout("policy", rng, HEADS[int(rng.integers(len(HEADS)))][0]) for _ in range(N)]
res["SP"] = summarise("S+P: scripted jump, then policy", SP)
print(f"   landed {sum(r['landed'] is not None for r in SP)}/{N}  ({time.time() - t0:.0f}s)")
SU = [rollout("noleft", rng, HEADS[int(rng.integers(len(HEADS)))][0]) for _ in range(N)]
res["SU"] = summarise("S+U: scripted jump, then random no-left", SU)
print(f"   ({time.time() - t0:.0f}s)")

# ---- upper bound for the jump under R0 and the break-even gamma (shortest head)
head = min(HEADS, key=lambda h: len(h[0]))[0]
obs, info = goto_s0()
hr, hx, hfl = [], [], []
for a in head:
    obs, r, term, trunc, info = env.step(a)
    s = st(base, info)
    hr.append(float(r)); hx.append(s[0]); hfl.append(s[2])
L = len(head)
pr = res["P"]


def g_fail(g):
    return float(np.mean([disc(r["r"], g) for r in P]))


def g_upper(g, speed=0.4):
    return disc(hr, g) + g ** L * speed / (1 - g)


def breakeven(speed):
    lo, hi = 0.5, 0.9999
    if g_upper(lo, speed) > g_fail(lo):
        return lo
    for _ in range(60):
        mid = (lo + hi) / 2
        if g_upper(mid, speed) > g_fail(mid):
            hi = mid
        else:
            lo = mid
    return hi


ub = {str(g): (g_upper(g), g_fail(g)) for g in GAMMAS}
res["upper"] = dict(head_len=L, x_land=hx[-1], head_x=hx, head_r=hr, by_gamma=ub,
                    breakeven={str(sp): breakeven(sp) for sp in [0.4, 0.3, 0.2]})
print(f"\nUpper bound (R0): jump lands at x {hx[-1]} after {L} steps; head x {hx}")
print("   g: (jump then run 12 px/step forever, policy) " + "  ".join(f"g{g}=({u:.2f}, {f:.2f})" for g, (u, f) in ub.items()))
print("   break-even gamma for after-landing speed 12 / 9 / 6 px per step:", res["upper"]["breakeven"])

# ---- S+B: competent continuation found by random search from the landing state (snapshot on env2)
if SEARCH > 0:
    base2._restore()  # env2's snapshot slot holds s0 (its reset point is gone, so no goto_s0 on env2 here)
    base2.done = False
    for a in head:
        _, _, _, _, info = fs2.step(a)
    assert on_tall(*st(base2, info)), "head did not land on env2"
    x_land, time_land = st(base2, info)[0], int(info["time"])
    base2._backup()

    def cont(seq):
        base2._restore()
        base2.done = False
        x_prev, t_prev = x_land, time_land
        rs, xs, fls = [], [], []
        for a in seq:
            _, _, term, trunc, info = fs2.step(a)
            x, tm = int(info["x_pos"]), int(info["time"])
            r = ((x - x_prev) + min(0, tm - t_prev)) / 30.0
            died = term and not info.get("flag_get", False)
            if died:
                r += DEATH
            rs.append(r); xs.append(x); fls.append(st(base2, info)[2])
            x_prev, t_prev = x, tm
            if term or trunc:
                return rs, xs, fls, "death" if died else "flag"
        return rs, xs, fls, "alive"

    srng = np.random.default_rng(7)
    best, reach = None, []
    for i in range(SEARCH):
        seq = []
        while len(seq) < 120:
            seq += [int(srng.choice(6, p=[.05, .25, .25, .15, .2, .1]))] * int(srng.integers(1, 9))
        rs, xs, fls, end = cont(seq[:120])
        reach.append(max(xs))
        score = (end != "death", xs[-1])
        if best is None or score > best[0]:
            best = (score, rs, xs, fls, end)
    reach = np.array(reach)
    _, rs_b, xs_b, fls_b, end_b = best
    full = variants(hr + rs_b, hx + xs_b, hfl + fls_b, end_b == "death", S0[0])
    res["SB"] = dict(end=end_b, x_end=xs_b[-1], steps=L + len(rs_b),
                     G={f"{rule}|{g}": disc(full[rule], g) for rule in RULES for g in GAMMAS},
                     p_reach={str(t): float(np.mean(reach > t)) for t in [783, 850, 900, 963, 1010, 1100]})
    print(f"\n== S+B: best of {SEARCH} continuations: end {end_b}, x {xs_b[-1]} after {L + len(rs_b)} steps "
          f"(window only); reach probs {res['SB']['p_reach']}")
    for rule in RULES:
        print(f"   {rule:5s}: " + "  ".join(f"g{g}={res['SB']['G'][f'{rule}|{g}']:6.2f}" for g in GAMMAS))

print("\n== Advantage of the correct jump: E[G|branch] - E[G|P] ==")
for name in ["SP", "SU", "SB"]:
    if not res.get(name):
        continue
    for rule in RULES:
        row = []
        for g in GAMMAS:
            v = res[name]["G"][f"{rule}|{g}"]
            v = v[0] if isinstance(v, (list, tuple)) else v
            row.append(f"g{g}={v - res['P']['G'][f'{rule}|{g}'][0]:+6.2f}")
        print(f"   {name:2s} {rule:5s}: " + "  ".join(row))
print(f"total {time.time() - t0:.0f}s")
if OUT:
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
