"""Save what the network sees at s0 (the decision state on the low tree of 1-3): the 4 stacked 84x84 frames,
plus the full-colour screen, side by side. Same prefix rule as experiments/jump_returns.py.

    python experiments/decision_view.py <ckpt> docs/reports/figures/1-3-decision-view.png
"""
import sys

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/d/mario-ppo")
from mario_ppo.env import make_env
from mario_ppo.evaluate import load_model

CK, OUT = sys.argv[1], sys.argv[2]
model, ck = load_model(CK)
model.eval()
cfg = dict(ck["config"]["env"])
cfg.update(eval_mode=False, noop_max=0, normalize_reward=False)
env = make_env("1-3", **cfg)
base = env.unwrapped
NAMES = ["NOOP", "R", "R+A", "R+B", "R+A+B", "A", "L"]
for s in range(100):
    rng = np.random.default_rng(s)
    obs, info = env.reset(seed=0)
    found = False
    for t in range(300):
        x, fl, yp = int(info.get("x_pos", 0)), int(base.ram[0x1D]), int(base.ram[0xB5])
        if t and 590 <= x <= 625 and fl == 0 and yp == 1:
            found = True
            break
        with torch.no_grad():
            logits, v = model(torch.as_tensor(np.asarray(obs)[None]))
        p = torch.softmax(logits, -1)[0].numpy().astype(np.float64)
        obs, _, term, trunc, info = env.step(int(rng.choice(7, p=p / p.sum())))
        if term or trunc:
            break
    if found:
        break
with torch.no_grad():
    logits, v = model(torch.as_tensor(np.asarray(obs)[None]))
p = torch.softmax(logits, -1)[0].numpy()
print(f"s0 x {x}; V {float(v):.3f}; policy " + " ".join(f"{n}:{q:.3f}" for n, q in zip(NAMES, p)))
stack = np.asarray(obs)  # (4, 84, 84)
tiles = [np.asarray(Image.fromarray(f).resize((240, 240), Image.NEAREST)) for f in stack]
gray = np.concatenate([np.pad(t, ((0, 0), (0, 6)), constant_values=255) for t in tiles], axis=1)
rgb = np.asarray(Image.fromarray(base.screen.copy()).resize((256, 240)))
gray_rgb = np.repeat(gray[..., None], 3, axis=2)
row = np.concatenate([rgb, np.full((240, 10, 3), 255, np.uint8), gray_rgb], axis=1)
iio.imwrite(OUT, row)
print("saved", OUT, row.shape)
