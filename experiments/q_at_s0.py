"""Diagnosis only (no training): Q(s0, a) for each first action at the 1-3 decision state, under a checkpoint's own
training reward and gamma (docs/optimize-1-1.md, Mục 19-20).

s0 = Mario first standing on the low tree (x 590-625), reached by deterministic replay of a sampled prefix of the
checkpoint's own policy in its training env (its env config, including tricks such as pit_fall_claw; noop_max 0).
For each first action a: N rollouts that force a at s0, then sample the policy. Reports pi(a|s0), Q(s0,a) = mean
discounted return, the advantage Q - V, the share that lands on the tall tree, and the share that passes x 783.
Also the scripted correct jump (A, then R+A, R until landed) followed by the policy, for reference.

    python experiments/q_at_s0.py <ckpt> [N=40]
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/d/mario-ppo")
from mario_ppo.env import make_env
from mario_ppo.evaluate import load_model

CK = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
NAMES = ["NOOP", "R", "R+A", "R+B", "R+A+B", "A", "L"]
torch.set_num_threads(3)
model, ck = load_model(CK)
model.eval()
GAMMA = float(ck["config"]["algo"]["gamma"])
cfg = dict(ck["config"]["env"])
cfg.update(eval_mode=False, noop_max=0, normalize_reward=False)
env = make_env("1-3", **cfg)
ram = env.unwrapped.ram


def st(info):
    return int(info.get("x_pos", 0)), int(ram[0xCE]), int(ram[0x1D]), int(ram[0xB5])


def on_tall(x, y, fl, yp):
    return 662 <= x <= 740 and fl == 0 and yp == 1 and y < 80


def pol(obs):
    with torch.no_grad():
        logits, v = model(torch.as_tensor(np.asarray(obs)[None]))
    p = torch.softmax(logits, -1)[0].numpy().astype(np.float64)
    return p / p.sum(), float(v.reshape(-1)[0])


prefix = None
for s in range(100):
    rng = np.random.default_rng(s)
    obs, info = env.reset(seed=0)
    acts = []
    for _ in range(300):
        x, y, fl, yp = st(info)
        if acts and 590 <= x <= 625 and fl == 0 and yp == 1:
            prefix = acts
            break
        p, _ = pol(obs)
        a = int(rng.choice(7, p=p))
        obs, _, term, trunc, info = env.step(a)
        acts.append(a)
        if term or trunc:
            break
    if prefix:
        break
if not prefix:
    sys.exit("policy never stands on the low tree")


def goto():
    obs, info = env.reset(seed=0)
    for a in prefix:
        obs, _, _, _, info = env.step(a)
    return obs, info


obs0, info0 = goto()
p0, v0 = pol(obs0)
print(f"ckpt {CK}  gamma {GAMMA}  claw {cfg.get('pit_fall_claw', False)}  s0 x {st(info0)[0]}  V(s0) {v0:.3f}")
print("pi(.|s0): " + " ".join(f"{n}:{q:.3f}" for n, q in zip(NAMES, p0)))


def run(first, rng, max_steps=400):
    obs, info = goto()
    G, disc, landed, xmax = 0.0, 1.0, False, 0
    for t in range(max_steps):
        if isinstance(first, list) and not landed and t < len(first):
            a = first[t]
        elif isinstance(first, int) and t == 0:
            a = first
        else:
            p, _ = pol(obs)
            a = int(rng.choice(7, p=p))
        obs, r, term, trunc, info = env.step(a)
        G += disc * r
        disc *= GAMMA
        s = st(info)
        xmax = max(xmax, s[0])
        landed = landed or on_tall(*s)
        if term or trunc:
            break
    return G, landed, xmax > 783


rng = np.random.default_rng(1)
rows = []
for a in range(7):
    res = [run(a, rng) for _ in range(N)]
    G = np.array([r[0] for r in res])
    rows.append((NAMES[a], p0[a], G.mean(), G.std() / np.sqrt(N), G.mean() - v0,
                 np.mean([r[1] for r in res]), np.mean([r[2] for r in res])))
res = [run([5] + [2] * k + [1] * 15, rng) for k in rng.integers(6, 12, size=N)]
G = np.array([r[0] for r in res])
rows.append(("jump*", float("nan"), G.mean(), G.std() / np.sqrt(N), G.mean() - v0,
             np.mean([r[1] for r in res]), np.mean([r[2] for r in res])))
print(f"{'first':>6} {'pi':>6} {'Q':>7} {'se':>5} {'Q-V':>7} {'landed':>7} {'x>783':>6}")
for n, p, q, se, adv, l, past in rows:
    print(f"{n:>6} {p:>6.3f} {q:>7.3f} {se:>5.2f} {adv:>+7.3f} {l:>7.2f} {past:>6.2f}")
print("jump* = scripted A then R+A 6-11 steps then R until landed (lands only if s0 is x 597-like), then the policy")
