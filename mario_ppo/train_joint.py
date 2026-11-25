"""Huấn luyện MỘT mạng trên NHIỀU màn cùng lúc (tệp thêm của nhóm; train.py của thầy giữ nguyên).

    python -m mario_ppo.train_joint --config configs/group-1-1-and-4-1.yaml run_dir=runs/1-1+4-1/from-scratch

Khác train.py ở đúng năm chỗ, phần còn lại (thu dữ liệu, GAE, ppo_update, lịch tốc độ học, checkpoint) chép theo
train.py:
  1. Cấu hình có khóa `levels` (danh sách màn) thay cho `level`. Môi trường thứ i chơi màn levels[i % số màn],
     nên mỗi màn nhận phần bước bằng nhau và MỖI lần cập nhật PPO đều có dữ liệu của mọi màn. Lợi thế (GAE)
     vẫn tính riêng theo từng môi trường, sau đó các mẫu được trộn khi chia mini-batch.
  2. Nhật ký tách theo màn: progress.csv có thêm các cột ep_return_<màn>, ep_length_<màn>, x_pos_<màn>,
     flag_rate_<màn>, episodes_<màn> (100 lượt gần nhất của riêng màn đó). Các cột chung ep_return, ep_length,
     x_pos, flag_rate là trung bình cộng của các màn, để màn có lượt ngắn không lấn át con số chung.
  3. Mỗi checkpoint được đánh giá trên từng màn: evals.csv có một dòng cho mỗi (checkpoint, màn),
     video là videos/step_*_<màn>.mp4.
  4. Khóa tùy chọn `stop_at_step` (cấp cao nhất của cấu hình, mặc định không có): lượt chạy dừng sau đoạn thu đầu
     tiên đưa bộ đếm bước tới mốc này, lưu checkpoint ở đúng bước đó rồi thoát. Lịch tốc độ học và lịch entropy
     vẫn tính theo algo.total_steps, nên chạy tiếp bằng `resume=<checkpoint đó>` đi đúng lịch của lượt đầy đủ.
     Dùng để chạy một đoạn có số bước định trước mà không phải dừng tay, ví dụ:
         python -m mario_ppo.train_joint --config <cấu hình> run_dir=<thư mục> stop_at_step=2000000
     Mỗi đoạn thu là num_envs * num_steps bước, nên bước dừng thật là bội số đầu tiên của số đó không nhỏ hơn mốc.
  5. Mục tùy chọn `allocation` (cấp cao nhất của cấu hình, mặc định không có = chia đều như mục 1): dồn bước cho
     màn còn trượt nhiều. Mỗi môi trường chơi được mọi màn (LevelSwitchEnv) và chọn màn cho từng lượt chơi mới;
     sau mỗi đoạn thu, phần bước mục tiêu của từng màn được tính lại từ tỉ lệ trượt của 100 lượt gần nhất:
         allocation:
           floor: 0.06        # phần bước tối thiểu của mỗi màn, để màn đã thạo không bị quên
           cap: 0.30          # phần bước tối đa của một màn, để một màn quá khó không nuốt hết ngân sách
           min_episodes: 20   # màn có ít lượt hơn số này trong cửa sổ thì coi như tỉ lệ trượt bằng 1
     Phần còn lại sau khi trừ sàn (1 - số màn * floor) chia theo tỉ lệ trượt. Số bước thật và phần mục tiêu của
     từng đoạn thu được ghi vào allocation.csv; progress.csv giữ nguyên các cột.

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


def level_shares(fail_rates, floor, cap):
    """Phần bước mục tiêu của từng màn (tổng bằng 1) theo tỉ lệ trượt `fail_rates`.

    Mỗi màn nhận `floor`; phần còn lại chia theo tỉ lệ trượt; không màn nào vượt `cap` (phần vượt trần chia lại
    cho các màn còn dưới trần, vẫn theo tỉ lệ trượt). Khi các màn còn dưới trần đều không trượt thì chia đều.
    """
    fail = np.clip(np.asarray(fail_rates, dtype=float), 0.0, 1.0)
    n = len(fail)
    if not (n and 0.0 <= floor <= cap and n * floor <= 1.0 + 1e-9 and n * cap >= 1.0 - 1e-9):
        raise ValueError(f"allocation: cần 0 <= floor <= cap, {n} * floor <= 1 <= {n} * cap; "
                         f"nhận floor={floor}, cap={cap}")
    share = np.full(n, float(floor))
    free = np.ones(n, dtype=bool)  # màn còn dưới trần
    rest = 1.0 - n * floor         # phần chưa chia
    while rest > 1e-12 and free.any():
        weight = np.where(free, fail, 0.0)
        if weight.sum() <= 0.0:
            weight = free.astype(float)
        add = rest * weight / weight.sum()
        over = free & (share + add > cap + 1e-12)
        if not over.any():
            share += add
            break
        rest -= float((cap - share[over]).sum())
        share[over] = cap
        free &= ~over
    return share


def episode_probs(shares, mean_lengths):
    """Xác suất chọn màn cho lượt chơi kế tiếp để phần BƯỚC của mỗi màn bằng `shares`.

    Màn có lượt dài cần ít lượt hơn để nhận cùng số bước, nên xác suất tỉ lệ với shares / độ dài lượt trung bình.
    Màn chưa có lượt nào (độ dài nan) dùng trung bình độ dài của các màn đã có.
    """
    shares, lengths = np.asarray(shares, dtype=float), np.asarray(mean_lengths, dtype=float)
    known = np.isfinite(lengths) & (lengths > 0)
    lengths = np.where(known, lengths, lengths[known].mean() if known.any() else 1.0)
    probs = shares / lengths
    return probs / probs.sum()


class LevelSwitchEnv(gym.Env):
    """Một môi trường chơi được nhiều màn; chỉ dùng khi huấn luyện với mục `allocation`.

    Mỗi lần reset, môi trường chọn màn cho lượt chơi kế tiếp theo xác suất `level_probs` (tiến trình huấn luyện đặt
    lại sau mỗi đoạn thu bằng envs.set_attr); khi `level_probs` còn là None thì giữ màn đang chơi. Mỗi màn là một
    môi trường make_env riêng, tạo khi cần lần đầu. Quan sát vẫn chỉ là khung hình: chỉ số màn nằm trong
    info["level_index"] để tiến trình huấn luyện ghi nhật ký, mạng không nhận được nó (Quy định 3.5).
    """

    def __init__(self, levels, start_level, seed, env_cfg):
        self.levels = [str(level) for level in levels]
        self.level_probs = None
        self._seed, self._env_cfg, self._envs = int(seed), dict(env_cfg), {}
        self._index = self.levels.index(str(start_level))
        self._pick = np.random.default_rng(self._seed)
        env = self._current()
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.metadata, self.render_mode = dict(env.metadata), env.render_mode

    def _current(self):
        if self._index not in self._envs:
            from mario_ppo.env import make_env
            self._envs[self._index] = make_env(self.levels[self._index], seed=self._seed + 1000 * self._index,
                                               **self._env_cfg)
        return self._envs[self._index]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._pick = np.random.default_rng(seed)
        if self.level_probs is not None:
            probs = np.asarray(self.level_probs, dtype=float)
            self._index = int(self._pick.choice(len(self.levels), p=probs / probs.sum()))
        obs, info = self._current().reset(seed=seed, options=options)
        return obs, dict(info, level_index=self._index)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._current().step(action)
        return obs, reward, terminated, truncated, dict(info, level_index=self._index)

    def close(self):
        for env in self._envs.values():
            env.close()
        self._envs = {}


def _make_vec_env(env_cfg, env_levels, seed, switch_levels=None):
    """`switch_levels` (danh sách màn) bật LevelSwitchEnv: môi trường i bắt đầu ở env_levels[i] rồi đổi màn theo lượt."""
    from mario_ppo.env import make_env
    if switch_levels is None:
        fns = [functools.partial(make_env, level, seed=seed + i, **env_cfg) for i, level in enumerate(env_levels)]
    else:
        fns = [functools.partial(LevelSwitchEnv, list(switch_levels), level, seed + i, env_cfg)
               for i, level in enumerate(env_levels)]
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
    stop_at = cfg.get("stop_at_step")  # mốc dừng sớm của nhóm; None = chạy tới algo.total_steps như train.py
    if stop_at is not None and int(stop_at) <= global_step:
        raise ValueError(f"stop_at_step={stop_at} phải lớn hơn bước hiện tại ({global_step})")
    last_step = a.total_steps if stop_at is None else min(a.total_steps, int(stop_at))
    alloc = cfg.get("allocation")  # dồn bước theo tỉ lệ trượt (mục 5 ở đầu tệp); None = mỗi môi trường một màn cố định
    if alloc is not None:
        level_shares(np.ones(len(levels)), alloc.floor, alloc.cap)  # báo lỗi cấu hình trước khi mở môi trường

    progress, evals = _CSV(run / "progress.csv"), _CSV(run / "evals.csv")
    allocation = _CSV(run / "allocation.csv")
    envs = _make_vec_env(env_cfg, env_levels, cfg.seed, switch_levels=levels if alloc is not None else None)
    playing = np.array([levels.index(level) for level in env_levels])  # chỉ số màn mỗi môi trường đang chơi
    level_steps = np.zeros(len(levels), dtype=np.int64)                # số bước của từng màn trong đoạn thu hiện tại
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
        while global_step < last_step:
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
                level_steps += np.bincount(playing, minlength=len(levels))
                done = np.logical_or(term, trunc)
                if done.any():
                    fin = info["final_info"]
                    for i in np.flatnonzero(done):
                        level = levels[playing[i]]
                        played = (int(fin["world"][i]), int(fin["stage"][i]))
                        if played != parse_level(level):  # tự kiểm tra: môi trường i phải chơi đúng màn đang ghi cho nó
                            raise RuntimeError(f"môi trường {i} được ghi là màn {level} nhưng đang chơi {played}")
                        recent[level].append((float(fin["episode"]["r"][i]), int(fin["episode"]["l"][i]),
                                              int(fin["x_pos"][i]), bool(fin["flag_get"][i])))
                    cut = np.flatnonzero(np.logical_and(trunc, ~term))
                    if len(cut):  # giá trị của quan sát cuối của lượt bị cắt (không phải quan sát đầu lượt mới)
                        final_obs = torch.as_tensor(np.stack([info["final_obs"][i] for i in cut]))
                        _, _, v_fin = model.act(final_obs)
                        b_final[t, cut] = v_fin
                if alloc is not None:  # môi trường vừa hết lượt có thể đã sang màn khác
                    playing = np.asarray(info["level_index"], dtype=np.int64)
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

            # ---- dồn bước cho màn còn trượt: đặt xác suất chọn màn cho các lượt chơi kế tiếp ----
            if alloc is not None:
                min_episodes = alloc.get("min_episodes", 20)
                fail = [1.0 if len(recent[level]) < min_episodes else 1.0 - per_level[level][3] for level in levels]
                shares = level_shares(fail, alloc.floor, alloc.cap)
                probs = episode_probs(shares, [per_level[level][1] for level in levels])
                envs.set_attr("level_probs", [tuple(float(p) for p in probs)] * N)
                # steps_*: số bước thật của đoạn thu vừa xong; share_*, prob_*: mục tiêu cho các lượt chơi sắp tới
                allocation.write({"global_step": global_step,
                                  **{f"steps_{level}": int(level_steps[k]) for k, level in enumerate(levels)},
                                  **{f"share_{level}": round(float(shares[k]), 4) for k, level in enumerate(levels)},
                                  **{f"prob_{level}": round(float(probs[k]), 4) for k, level in enumerate(levels)}})
            level_steps[:] = 0
            if global_step >= next_ckpt:
                checkpoint()
                next_ckpt += cfg.checkpoint_every
        if not list((run / "checkpoints").glob(f"step_{global_step:09d}.pt")):
            checkpoint()
        if global_step < a.total_steps:
            print(f"stopped at step {global_step} (stop_at_step={stop_at}, total_steps={a.total_steps}); "
                  f"continue with resume={run / 'checkpoints' / f'step_{global_step:09d}.pt'}", flush=True)
    finally:
        envs.close()
        progress.close()
        evals.close()
        allocation.close()
    return run


def load_config(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/group-1-1-and-4-1.yaml")
    p.add_argument("overrides", nargs="*", help="khóa=giá trị, ví dụ algo.total_steps=3000000 run_dir=runs/thu")
    args = p.parse_args(argv)
    return OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))


if __name__ == "__main__":
    print(f"run dir: {train(load_config())}")
