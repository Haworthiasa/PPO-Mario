"""Check PitFallClaw (mario_ppo/wrappers.py) on the real emulator before a run uses it (docs/optimize-1-1.md, Mục 19).

Two training envs step with the same actions: one with env.pit_fall_claw, one without. Per death it compares the
wrapper's claw with the claw computed here from the RAM float state (0x001D == 0 at the end of a step = on the
ground), and checks that the two rewards differ by exactly claw / 30 at the death step and nowhere else.
It also reports how well an info-only "on the ground" signal (y_pos unchanged since the previous agent step) agrees
with RAM; that signal was the first version of the wrapper and was dropped because it missed short landings.

    python experiments/pit_claw_check.py [level=1-3] [episodes=60] [ckpt]
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/d/mario-ppo")
from mario_ppo.env import make_env
from mario_ppo.evaluate import load_model

LEVEL = sys.argv[1] if len(sys.argv) > 1 else "1-3"
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 60
CK = sys.argv[3] if len(sys.argv) > 3 else "runs/1-3/tuning/first-819k/checkpoints/step_000819200.pt"
model, ck = load_model(CK)
model.eval()
cfg = dict(ck["config"]["env"])
cfg.update(eval_mode=False, noop_max=0, normalize_reward=False)
with_claw = make_env(LEVEL, **dict(cfg, pit_fall_claw=True))
plain = make_env(LEVEL, **cfg)
ram = with_claw.unwrapped.ram

agree = np.zeros((2, 2), dtype=int)  # rows: RAM on ground (0/1), cols: wrapper on ground (0/1); death steps excluded
claws, mism, other_diff, flags = [], 0, 0, 0
rng = np.random.default_rng(0)
for ep in range(EPISODES):
    mode = ["policy", "noleft", "uniform"][ep % 3]
    obs, info = with_claw.reset(seed=ep)
    plain.reset(seed=ep)
    prev_y, ram_ground_x, done, t = None, None, False, 0
    while not done and t < 2000:
        if mode == "policy":
            with torch.no_grad():
                p = torch.softmax(model(torch.as_tensor(np.asarray(obs)[None]))[0], -1)[0].numpy().astype(float)
            a = int(rng.choice(7, p=p / p.sum()))
        else:
            a = int(rng.integers(0, 6 if mode == "noleft" else 7))
        obs, r1, term, trunc, info = with_claw.step(a)
        _, r0, term0, trunc0, info0 = plain.step(a)
        assert (term, trunc, info["x_pos"]) == (term0, trunc0, info0["x_pos"]), "the two envs drifted apart"
        x, y, fl = int(info["x_pos"]), int(info["y_pos"]), int(ram[0x1D])
        died = term and not info.get("flag_get", False)
        if fl == 0:
            ram_ground_x = x
        if not died:
            if prev_y is not None:
                agree[int(fl == 0), int(y == prev_y)] += 1
            if abs(r1 - r0) > 1e-5:
                other_diff += 1
        else:
            got = int(info.get("pit_fall_claw", 0))
            want = max(0, x - ram_ground_x) if ram_ground_x is not None else 0
            claws.append((mode, x, got, want))
            mism += int(abs(got - want) > 0)
            if abs((r0 - r1) - got / 30.0) > 1e-4:
                other_diff += 1
        flags += int(info.get("flag_get", False))
        prev_y, done, t = y, term or trunc, t + 1

print(f"{LEVEL}: {EPISODES} episodes, {agree.sum()} live steps, {len(claws)} deaths, {flags} flags")
print("info-only signal (y_pos unchanged) vs RAM, rows RAM air/ground, cols info air/ground:")
print(agree)
print(f"info says ground but RAM says air: {agree[0, 1]} steps; RAM ground but info air: {agree[1, 0]} steps")
print(f"deaths where wrapper claw != RAM claw: {mism} of {len(claws)}")
print(f"reward differences outside a death step, or not equal to claw/30 at it: {other_diff}")
for c in claws[:15]:
    print("   mode %-7s death x %5d  claw wrapper %4d px  RAM %4d px" % c)
big = [c for c in claws if abs(c[2] - c[3]) > 0]
for c in big[:15]:
    print("   MISMATCH mode %-7s death x %5d  wrapper %4d  RAM %4d" % c)
