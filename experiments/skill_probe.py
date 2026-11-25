"""Read-only policy diagnostics and synthetic architecture timing; no training run.

Run from the repository root with its existing Mario Python environment:
    python experiments/skill_probe.py --out runs/skill-analysis-20261121/probe.json
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mario_ppo.env import make_env
from mario_ppo.evaluate import load_model
from mario_ppo.model import ActorCritic
from mario_ppo.wrappers import ACTION_SETS


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(F.relu(x))))


class SmallImpala(nn.Module):
    """Timing candidates only; these randomly initialized models are not evaluated in game."""

    def __init__(self, regions=False):
        super().__init__()
        blocks = []
        previous = 4
        for channels in (16, 32, 32):
            blocks.extend((nn.Conv2d(previous, channels, 3, padding=1),
                           nn.MaxPool2d(3, stride=2, padding=1),
                           Residual(channels), Residual(channels)))
            previous = channels
        self.trunk = nn.Sequential(*blocks, nn.ReLU())
        self.regions = regions
        self.hidden = nn.Linear(32 * (17 if regions else 11 * 11), 256)
        self.policy = nn.Linear(256, 7)
        self.value = nn.Linear(256, 1)
        self.to(memory_format=torch.channels_last)

    def forward(self, obs):
        x = (obs.float() / 255).contiguous(memory_format=torch.channels_last)
        x = self.trunk(x)
        if self.regions:
            x = torch.cat((F.adaptive_avg_pool2d(x, (4, 4)).flatten(1),
                           F.adaptive_avg_pool2d(x, 1).flatten(1)), 1)
        else:
            x = x.flatten(1)
        x = F.relu(self.hidden(x))
        return self.policy(x), self.value(x).squeeze(-1)


def architecture_probe():
    result = []
    for name, model in (("nature_current", ActorCritic(7)),
                        ("impala_16_32_32_flat256", SmallImpala()),
                        ("impala_16_32_32_regions4_global256", SmallImpala(True))):
        model.eval()
        inference = {}
        for batch in (1, 8):
            obs = torch.randint(0, 256, (batch, 4, 84, 84), dtype=torch.uint8)
            with torch.no_grad():
                for _ in range(5):
                    model(obs)
                samples = []
                for _ in range(3):
                    start = time.perf_counter()
                    for _ in range(15):
                        model(obs)
                    samples.append((time.perf_counter() - start) / 15 * 1000)
            inference[str(batch)] = round(statistics.median(samples), 3)
        obs = torch.randint(0, 256, (256, 4, 84, 84), dtype=torch.uint8)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        samples = []
        model.train()
        for i in range(5):
            start = time.perf_counter()
            logits, value = model(obs)
            loss = logits.square().mean() + value.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i >= 2:
                samples.append((time.perf_counter() - start) * 1000)
        result.append({"name": name, "parameters": sum(p.numel() for p in model.parameters()),
                       "inference_ms_by_batch": inference,
                       "synthetic_forward_backward_adam_batch256_ms": round(statistics.median(samples), 3)})
        print("architecture", result[-1], flush=True)
    return result


def jump_probe():
    actions = ACTION_SETS["simple"]
    lookup = lambda buttons: next(i for i, combo in enumerate(actions) if set(combo) == set(buttons))
    rows = []
    steps_total = 0
    env = make_env("1-1", eval_mode=True, normalize_reward=False)
    try:
        for movement, warmup in (([], 0), (["right"], 5), (["right", "B"], 5)):
            for hold in (1, 2, 4, 8, 12):
                obs, info = env.reset(seed=2026)
                # Let Mario reach the floor before measuring, without changing game state directly.
                for _ in range(12):
                    obs, _, term, trunc, info = env.step(0)
                    steps_total += 1
                    assert not (term or trunc)
                for _ in range(warmup):
                    obs, _, term, trunc, info = env.step(lookup(movement))
                    steps_total += 1
                    assert not (term or trunc)
                x0, y0 = int(info["x_pos"]), int(info["y_pos"])
                trajectory = []
                ended = False
                for t in range(45):
                    buttons = movement + (["A"] if t < hold else [])
                    action = lookup(buttons if buttons else ["NOOP"])
                    obs, _, term, trunc, info = env.step(action)
                    steps_total += 1
                    trajectory.append((int(info["x_pos"]), int(info["y_pos"])))
                    ended = bool(term or trunc)
                    if ended or (t > 0 and any(y > y0 + 2 for _, y in trajectory) and info["y_pos"] <= y0):
                        break
                rows.append({"movement": "+".join(movement) or "stationary", "hold_A_agent_steps": hold,
                             "start_x": x0, "start_y": y0,
                             "apex_height_pixels": max(y for _, y in trajectory) - y0,
                             "until_landing_or_end_agent_steps": len(trajectory),
                             "distance_pixels": trajectory[-1][0] - x0, "terminated_or_truncated": ended})
        print("jumps", rows, flush=True)
    finally:
        env.close()
    return {"rows": rows, "environment_agent_steps_including_settle": steps_total}


def policy_probe():
    checkpoint = ROOT / "runs/1-1+4-1/from-scratch/checkpoints/step_002904064.pt"
    model, ck = load_model(checkpoint)
    model.eval()
    result = []
    step_count = 0
    for level in ("1-1", "4-1", "3-2", "1-3", "2-3", "6-1"):
        env = make_env(level, **dict(ck["config"]["env"], eval_mode=True))
        observations = []
        try:
            for seed in (7, 11, 101):
                torch.manual_seed(seed)
                obs, _ = env.reset(seed=seed)
                for _ in range(128):
                    observations.append(np.asarray(obs).copy())
                    action, _, _ = model.act(torch.as_tensor(np.asarray(obs)[None]))
                    obs, _, term, trunc, info = env.step(int(action))
                    step_count += 1
                    if term or trunc:
                        break
            obs_tensor = torch.as_tensor(np.stack(observations))
            with torch.no_grad():
                logits, _ = model(obs_tensor)
                base = torch.distributions.Categorical(logits=logits)
                row = {"level": level, "observations": len(observations),
                       "entropy_mean": round(base.entropy().mean().item(), 5),
                       "mean_jump_button_probability": round(base.probs[:, [2, 4, 5]].sum(-1).mean().item(), 5),
                       "perturbations": {}}
                for name in ("repeat_latest_frame", "hide_top_10_rows", "shift_right_2_pixels"):
                    changed = obs_tensor.clone()
                    if name == "repeat_latest_frame":
                        changed = changed[:, -1:].repeat(1, 4, 1, 1)
                    elif name == "hide_top_10_rows":
                        changed[:, :, :10] = 0
                    else:
                        changed[:, :, :, 2:] = obs_tensor[:, :, :, :-2]
                        changed[:, :, :, :2] = obs_tensor[:, :, :, :1]
                    other_logits, _ = model(changed)
                    other = torch.distributions.Categorical(logits=other_logits)
                    row["perturbations"][name] = {
                        "mean_KL_original_to_changed": round(torch.distributions.kl_divergence(base, other).mean().item(), 5),
                        "argmax_action_agreement": round((logits.argmax(-1) == other_logits.argmax(-1)).float().mean().item(), 5)}
                result.append(row)
                print("policy", row, flush=True)
        finally:
            env.close()
    return {"checkpoint": str(checkpoint.relative_to(ROOT)), "rows": result,
            "environment_agent_steps": step_count,
            "interpretation": "Sensitivity diagnostics on existing-policy trajectories, not success-rate evaluations or proof of causality."}


def log_probe():
    path = ROOT / "runs/multi-level/restart-test/progress.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    updates = [float(r["minibatch_updates"]) for r in rows]
    return {"source": str(path.relative_to(ROOT)), "updates": len(rows),
            "fraction_stopped_before_160_minibatches": sum(v < 160 for v in updates) / len(rows),
            "minibatch_updates_min_median_max": [min(updates), statistics.median(updates), max(updates)],
            "KL_mean_max": [statistics.mean(float(r["approx_kl"]) for r in rows),
                            max(float(r["approx_kl"]) for r in rows)],
            "warning": "Aggregate KL can hide per-level drift; learning-rate and entropy both changed on resume."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.manual_seed(2026)
    np.random.seed(2026)
    start = time.perf_counter()
    result = {"scope": "Diagnostics only; zero PPO training environment steps; no checkpoint writes.",
              "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
              "architectures": architecture_probe(), "jumps": jump_probe(), "policy": policy_probe(),
              "restart_log": log_probe(),
              "discount": {str(g): {str(k): g**k for k in (10, 20, 30, 50, 100)} for g in (0.9, 0.97, 0.99)}}
    result["elapsed_seconds"] = round(time.perf_counter() - start, 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved", args.out, "seconds", result["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
