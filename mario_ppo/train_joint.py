"""Huấn luyện MỘT mạng trên NHIỀU màn cùng lúc (tệp thêm của nhóm; train.py của thầy giữ nguyên).

    python -m mario_ppo.train_joint --config configs/group-1-1-and-4-1.yaml run_dir=runs/1-1+4-1/from-scratch

Khác train.py ở đúng ba chỗ, phần còn lại (thu dữ liệu, GAE, ppo_update, lịch tốc độ học, checkpoint) chép theo
train.py:
  1. Cấu hình có khóa `levels` (danh sách màn) thay cho `level`. Môi trường thứ i chơi màn levels[i % số màn],
     nên mỗi màn nhận phần bước bằng nhau và MỖI lần cập nhật PPO đều có dữ liệu của mọi màn. Lợi thế (GAE)
     vẫn tính riêng theo từng môi trường, sau đó các mẫu được trộn khi chia mini-batch.
  2. Nhật ký tách theo màn: progress.csv có thêm các cột ep_return_<màn>, ep_length_<màn>, x_pos_<màn>,
     flag_rate_<màn>, episodes_<màn> (100 lượt gần nhất của riêng màn đó). Các cột chung ep_return, ep_length,
     x_pos, flag_rate là trung bình cộng của các màn, để màn có lượt ngắn không lấn át con số chung.
  3. Mỗi checkpoint được đánh giá trên từng màn: evals.csv có một dòng cho mỗi (checkpoint, màn),
     video là videos/step_*_<màn>.mp4.

Mạng không được báo đang ở màn nào: đầu vào vẫn chỉ là khung hình (Quy định 3.5). `info` chỉ dùng để ghi nhật ký.
Với `levels` chỉ có một màn, tệp này cho đúng các con số của train.py (bài kiểm tra tests_group/test_train_joint.py).

Chấm một checkpoint của tệp này bằng evaluate.py nguyên bản, mỗi màn một lần, luôn kèm --level:
    python -m mario_ppo.evaluate --ckpt <ckpt> --level 1-1 --episodes 20 --seed 2026
    python -m mario_ppo.evaluate --ckpt <ckpt> --level 4-1 --episodes 20 --seed 2026
"""
import argparse
import functools
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf

from mario_ppo.train import _CSV, _device

# Không import torch ở mức module, cùng lý do với train.py: các tiến trình môi trường (forkserver) nạp lại module chính.


def assign_levels(levels, num_envs):
    """Màn của từng môi trường: môi trường thứ i chơi levels[i % len(levels)]."""
    from mario_ppo.env import parse_level
    levels = [str(level) for level in levels]
    if not levels or len(set(levels)) != len(levels):
        raise ValueError(f"levels phải là danh sách màn không rỗng, không trùng nhau: {levels!r}")
    for level in levels:
        parse_level(level)
    if num_envs % len(levels):
        raise ValueError(f"algo.num_envs={num_envs} phải chia hết cho số màn ({len(levels)}) "
                         "để các màn nhận số bước bằng nhau")
    return [levels[i % len(levels)] for i in range(num_envs)]


def _make_vec_env(env_cfg, env_levels, seed):
    from mario_ppo.env import make_env
    fns = [functools.partial(make_env, level, seed=seed + i, **env_cfg) for i, level in enumerate(env_levels)]
    mode = gym.vector.AutoresetMode.SAME_STEP  # như train.py
    if len(fns) == 1:
        return gym.vector.SyncVectorEnv(fns, autoreset_mode=mode)
    return gym.vector.AsyncVectorEnv(fns, context="forkserver", autoreset_mode=mode)


def train(cfg):
    """Huấn luyện theo cấu hình OmegaConf `cfg` (có khóa `levels`); trả đường dẫn run_dir."""
    import torch

    from mario_ppo.env import parse_level
    from mario_ppo.evaluate import run_episodes
    from mario_ppo.model import ActorCritic
    from mario_ppo.ppo import compute_gae, linear_schedule, ppo_update
    from mario_ppo.wrappers import ACTION_SETS

    a = cfg.algo
    env_levels = assign_levels(cfg.levels, a.num_envs)
    levels = env_levels[: len(cfg.levels)]
    cfg = OmegaConf.merge(cfg, {"level": "+".join(levels)})  # nhãn để hiển thị; đánh giá phải kèm --level
    run = Path(cfg.run_dir or f"runs/{cfg.level}/{datetime.now():%Y-%m-%d_%H-%M-%S}")
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "videos").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run / "config.yaml", resolve=True)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    env_cfg = dict(cfg_dict["env"], gamma=a.gamma)

    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _device(cfg.device)

    n_actions = len(ACTION_SETS[cfg.env.actions])
    model = ActorCritic(n_actions, in_channels=cfg.env.frame_stack, **cfg_dict.get("model", {})).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=a.adam_eps)
    global_step = 0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        global_step = ck["global_step"]
        print(f"resumed from {cfg.resume} at step {global_step}")

    progress, evals = _CSV(run / "progress.csv"), _CSV(run / "evals.csv")
    envs = _make_vec_env(env_cfg, env_levels, cfg.seed)
    T, N = a.num_steps, a.num_envs
    obs_shape = envs.single_observation_space.shape
    b_obs = torch.zeros((T, N, *obs_shape), dtype=torch.uint8)
    b_act = torch.zeros((T, N), dtype=torch.long)
    b_logp, b_val, b_rew, b_term, b_trunc, b_final = (torch.zeros((T, N)) for _ in range(6))
    # (lợi tức thô, độ dài, x_pos, chạm cờ) của 100 lượt gần nhất, riêng từng màn
    recent = {level: deque(maxlen=100) for level in levels}
    next_ckpt = (global_step // cfg.checkpoint_every + 1) * cfg.checkpoint_every

    def checkpoint():
        tag = f"step_{global_step:09d}"
        path = run / "checkpoints" / f"{tag}.pt"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "global_step": global_step, "config": cfg_dict}, path)
        if cfg.eval_episodes:
            for level in levels:
                res = run_episodes(model, level, cfg_dict["env"], episodes=cfg.eval_episodes,
                                   seed=cfg.seed * 1000 + global_step // cfg.checkpoint_every,
                                   max_steps=cfg.eval_max_steps,
                                   video=(run / "videos" / f"{tag}_{level}.mp4") if cfg.video else None)
                evals.write({"global_step": global_step, "level": level, "flag_rate": res["flag_rate"],
                             "x_pos": " ".join(map(str, res["x_pos"]))})
                print(f"[eval] step {global_step} level {level}: flag_rate {res['flag_rate']:.2f} "
                      f"x_pos {res['x_pos']}", flush=True)
        return path

    try:
        obs, _ = envs.reset(seed=cfg.seed)
        obs = torch.as_tensor(obs)
        t_start, step_start = time.perf_counter(), global_step
        while global_step < a.total_steps:
            lr = linear_schedule(a.lr, global_step, a.total_steps) if a.anneal_lr else a.lr
            for g in opt.param_groups:
                g["lr"] = lr

            # ---- thu một đoạn T bước trên N môi trường (như train.py) ----
            model.eval()
            b_final.zero_()
            for t in range(T):
                action, logp, value = model.act(obs)
                next_obs, reward, term, trunc, info = envs.step(action.numpy())
                b_obs[t], b_act[t], b_logp[t], b_val[t] = obs, action, logp, value
                b_rew[t] = torch.as_tensor(reward, dtype=torch.float32)
                b_term[t] = torch.as_tensor(term, dtype=torch.float32)
                b_trunc[t] = torch.as_tensor(np.logical_and(trunc, ~term), dtype=torch.float32)
                done = np.logical_or(term, trunc)
                if done.any():
                    fin = info["final_info"]
                    for i in np.flatnonzero(done):
                        played = (int(fin["world"][i]), int(fin["stage"][i]))
                        if played != parse_level(env_levels[i]):  # tự kiểm tra: môi trường i phải chơi đúng màn đã gán
                            raise RuntimeError(f"môi trường {i} được gán màn {env_levels[i]} nhưng đang chơi {played}")
                        recent[env_levels[i]].append((float(fin["episode"]["r"][i]), int(fin["episode"]["l"][i]),
                                                      int(fin["x_pos"][i]), bool(fin["flag_get"][i])))
                    cut = np.flatnonzero(np.logical_and(trunc, ~term))
                    if len(cut):  # giá trị của quan sát cuối của lượt bị cắt (không phải quan sát đầu lượt mới)
                        final_obs = torch.as_tensor(np.stack([info["final_obs"][i] for i in cut]))
                        _, _, v_fin = model.act(final_obs)
                        b_final[t, cut] = v_fin
                obs = torch.as_tensor(next_obs)
            global_step += T * N
            _, _, last_value = model.act(obs)

            # ---- lợi thế (riêng từng môi trường) + cập nhật trên mẫu trộn của mọi màn ----
            adv, ret = compute_gae(b_rew, b_val, b_term, b_trunc, b_final, last_value, a.gamma, a.gae_lambda)
            model.train()
            batch = {"obs": b_obs.reshape(T * N, *obs_shape), "actions": b_act.reshape(-1),
                     "old_logp": b_logp.reshape(-1), "old_values": b_val.reshape(-1),
                     "advantages": adv.reshape(-1), "returns": ret.reshape(-1)}
            batch = {k: v.to(device) for k, v in batch.items()}
            stats = ppo_update(model, opt, batch, a)

            # ---- ghi nhật ký ----
            sps = (global_step - step_start) / (time.perf_counter() - t_start)
            per_level = {level: np.array(recent[level], dtype=float).mean(axis=0) if recent[level]
                         else np.full(4, np.nan) for level in levels}
            seen = [per_level[level] for level in levels if recent[level]]
            mean = np.mean(seen, axis=0) if seen else np.full(4, np.nan)  # trung bình cộng các màn đã có lượt
            row = {"global_step": global_step, "sps": round(sps, 1), "lr": lr,
                   **{k: round(float(v), 5) for k, v in stats.items()},
                   "episodes": sum(len(recent[level]) for level in levels), "ep_return": round(float(mean[0]), 3),
                   "ep_length": round(float(mean[1]), 1), "x_pos": round(float(mean[2]), 1),
                   "flag_rate": round(float(mean[3]), 3)}
            for level in levels:
                m = per_level[level]
                row.update({f"episodes_{level}": len(recent[level]), f"ep_return_{level}": round(float(m[0]), 3),
                            f"ep_length_{level}": round(float(m[1]), 1), f"x_pos_{level}": round(float(m[2]), 1),
                            f"flag_rate_{level}": round(float(m[3]), 3)})
            progress.write(row)
            by_level = "".join(f"  | {level} x {row[f'x_pos_{level}']:7.1f} flag {row[f'flag_rate_{level}']:.2f}"
                               for level in levels)
            print(f"step {global_step:>9d}  sps {sps:6.1f}  return {row['ep_return']:8.2f}  "
                  f"x {row['x_pos']:7.1f}  flag {row['flag_rate']:.2f}  kl {row.get('approx_kl', 0):.4f}{by_level}",
                  flush=True)
            if global_step >= next_ckpt:
                checkpoint()
                next_ckpt += cfg.checkpoint_every
        if not list((run / "checkpoints").glob(f"step_{global_step:09d}.pt")):
            checkpoint()
    finally:
        envs.close()
        progress.close()
        evals.close()
    return run


def load_config(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/group-1-1-and-4-1.yaml")
    p.add_argument("overrides", nargs="*", help="khóa=giá trị, ví dụ algo.total_steps=3000000 run_dir=runs/thu")
    args = p.parse_args(argv)
    return OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))


if __name__ == "__main__":
    print(f"run dir: {train(load_config())}")
