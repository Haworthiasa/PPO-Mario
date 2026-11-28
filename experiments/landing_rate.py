"""Diagnosis only (no training): how often a checkpoint's sampled policy lands on the tall tree of 1-3 (x 662-740,
standing, upper page) in whole episodes from the level start, in its own training env (its tricks, noop_max as in
training). This is the exploration rate that PPO actually gets for the jump (docs/optimize-1-1.md, Mục 20).

Also: the arrival on the low tree (first step standing at x 575-640): x and speed (px per agent step).

    python experiments/landing_rate.py <ckpt> [episodes=200]
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/d/mario-ppo")
from mario_ppo.env import make_env
from mario_ppo.evaluate import load_model

CK = sys.argv[1]
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 200
torch.set_num_threads(3)
model, ck = load_model(CK)
model.eval()
cfg = dict(ck["config"]["env"], eval_mode=False, normalize_reward=False)
env = make_env("1-3", **cfg)
ram = env.unwrapped.ram
rng = np.random.default_rng(0)
landed, past, arrive_x, arrive_v, reach_low = 0, 0, [], [], 0
for ep in range(EP):
    obs, info = env.reset(seed=10_000 + ep)
    done, prev_x, got_low, got_tall, xmax = False, None, False, False, 0
    while not done:
        with torch.no_grad():
            p = torch.softmax(model(torch.as_tensor(np.asarray(obs)[None]))[0], -1)[0].numpy().astype(float)
        obs, _, term, trunc, info = env.step(int(rng.choice(7, p=p / p.sum())))
        done = term or trunc
        x, y, fl, yp = int(info["x_pos"]), int(ram[0xCE]), int(ram[0x1D]), int(ram[0xB5])
        xmax = max(xmax, x)
        if not got_low and 575 <= x <= 640 and fl == 0 and yp == 1 and y > 80:
            got_low = True
            arrive_x.append(x)
            arrive_v.append(x - prev_x if prev_x is not None else 0)
        if 662 <= x <= 740 and fl == 0 and yp == 1 and y < 80:
            got_tall = True
        prev_x = x
    reach_low += got_low
    landed += got_tall
    past += xmax > 783
print(f"{CK}: {EP} episodes; stood on the low tree {reach_low}; landed on the tall tree {landed} "
      f"({landed / EP:.3f}); max x past 783 {past}")
if arrive_x:
    print(f"   arrival on the low tree: x mean {np.mean(arrive_x):.0f} (q10/90 {np.percentile(arrive_x, 10):.0f}/"
          f"{np.percentile(arrive_x, 90):.0f}), speed px/step mean {np.mean(arrive_v):.1f} "
          f"(q10/90 {np.percentile(arrive_v, 10):.0f}/{np.percentile(arrive_v, 90):.0f})")
