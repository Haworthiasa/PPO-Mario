"""Đo thang nhiễu gradient (gradient noise scale) của PPO tại một checkpoint. Không cập nhật trọng số.

Theo McCandlish và cộng sự (2018), "An Empirical Model of Large-Batch Training", phụ lục A:
  với gradient trên mini-batch cỡ B, E|g_B|^2 = |G|^2 + tr(Σ)/B,
  nên từ hai cỡ B_small < B_big ước lượng được |G|^2 và tr(Σ), rồi B_simple = tr(Σ) / |G|^2.
B_simple là cỡ batch mà ở đó nhiễu và tín hiệu của gradient bằng nhau. Batch nhỏ hơn nhiều so với B_simple thì
mỗi bước gradient chủ yếu là nhiễu; lớn hơn nhiều thì tốn mẫu mà hướng đi gần như không tốt hơn.

Cách làm: chạy `--rollouts` đoạn thu, mỗi đoạn num_envs x num_steps bước với chính sách của checkpoint (môi trường
huấn luyện, có mẹo huấn luyện), tính GAE như train.py, rồi tại θ của checkpoint (tỉ số chính sách = 1) tính:
  - G_r: gradient trên cả đoạn thu (4.096 mẫu) của mất mát PPO đầy đủ, và của riêng phần chính sách;
  - g_b: gradient trên từng mini-batch 256 mẫu của cùng đoạn thu.
Hai ước lượng:
  - trong đoạn thu (mini-batch so với cả đoạn): nhiễu khi chia mini-batch, như lúc huấn luyện;
  - giữa các đoạn thu: nhiễu của cả một đoạn thu 4.096 mẫu, có tính tương quan theo thời gian của các mẫu.
Lợi thế được chuẩn hóa một lần trên cả đoạn thu, dùng chung cho mọi cỡ batch.

    python experiments/gradient_noise.py --ckpt runs/1-1/run-1/checkpoints/step_001200128.pt --rollouts 6
    python experiments/gradient_noise.py --ckpt <ckpt> --levels 1-1 2-1 3-1 3-2 4-1 5-1 5-2 6-1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--levels", nargs="*", default=None, help="màn của từng môi trường (mặc định: màn trong cấu hình)")
    p.add_argument("--rollouts", type=int, default=6)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default=None, help="tệp JSON kết quả")
    p.add_argument("--num-envs", type=int, default=None, help="đổi số môi trường (mặc định như cấu hình)")
    p.add_argument("--num-steps", type=int, default=None, help="đổi số bước mỗi đoạn thu (mặc định như cấu hình)")
    args = p.parse_args()

    import torch
    from mario_ppo.model import ActorCritic
    from mario_ppo.ppo import compute_gae
    from mario_ppo.train_joint import _make_vec_env, assign_levels
    from mario_ppo.wrappers import ACTION_SETS

    torch.set_num_threads(6)
    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    a = cfg["algo"]
    levels = args.levels or cfg.get("levels") or [cfg["level"]]
    N, T = args.num_envs or a["num_envs"], args.num_steps or a["num_steps"]
    # Giữ mini-batch 256 mẫu khi đổi cỡ đoạn thu, để ước lượng "trong đoạn" so được với cấu hình gốc.
    a = dict(a, num_minibatches=T * N // (a["num_envs"] * a["num_steps"] // a["num_minibatches"]))
    env_levels = assign_levels(levels, N) if N % len(levels) == 0 else [levels[i % len(levels)] for i in range(N)]
    env_cfg = dict(cfg["env"], gamma=a["gamma"])
    step = ck["global_step"]
    # Hệ số entropy đúng như lúc huấn luyện ở bước này (lịch tuyến tính của nhóm, nếu có).
    ent = a["ent_coef"]
    if a.get("ent_coef_final") is not None:
        ent = a["ent_coef"] + (a["ent_coef_final"] - a["ent_coef"]) * min(1.0, step / a["total_steps"])

    model = ActorCritic(len(ACTION_SETS[cfg["env"]["actions"]]), in_channels=cfg["env"]["frame_stack"],
                        **cfg.get("model", {}))
    model.load_state_dict(ck["model"])
    params = [q for q in model.parameters()]

    def flat_grad(loss, retain=False):
        grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain)
        return torch.cat([(g if g is not None else torch.zeros_like(q)).reshape(-1) for g, q in zip(grads, params)])

    def losses(obs, actions, adv, ret):
        logits, values = model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        # Tại θ cũ, tỉ số = 1 nên phép cắt không hoạt động; gradient của -min(r A, clip(r) A) bằng -A ∇log π.
        ratio = torch.exp(logp - logp.detach())
        policy = -(ratio * adv).mean()
        value = ((values - ret) ** 2).mean()
        total = policy + a["vf_coef"] * value - ent * dist.entropy().mean()
        return policy, total

    envs = _make_vec_env(env_cfg, env_levels, args.seed)
    obs_shape = envs.single_observation_space.shape
    b_obs = torch.zeros((T, N, *obs_shape), dtype=torch.uint8)
    b_act = torch.zeros((T, N), dtype=torch.long)
    b_val, b_rew, b_term, b_trunc, b_final = (torch.zeros((T, N)) for _ in range(5))

    rec = {"policy": {"G": [], "g_small_sq": []}, "total": {"G": [], "g_small_sq": []}}
    flags = []
    t0 = time.perf_counter()
    try:
        obs, _ = envs.reset(seed=args.seed)
        obs = torch.as_tensor(obs)
        model.eval()
        for r in range(args.rollouts):
            b_final.zero_()
            for t in range(T):
                action, _, value = model.act(obs)
                next_obs, reward, term, trunc, info = envs.step(action.numpy())
                b_obs[t], b_act[t], b_val[t] = obs, action, value
                b_rew[t] = torch.as_tensor(reward, dtype=torch.float32)
                b_term[t] = torch.as_tensor(term, dtype=torch.float32)
                b_trunc[t] = torch.as_tensor(np.logical_and(trunc, ~term), dtype=torch.float32)
                done = np.logical_or(term, trunc)
                if done.any():
                    fin = info["final_info"]
                    flags += [bool(fin["flag_get"][i]) for i in np.flatnonzero(done)]
                    cut = np.flatnonzero(np.logical_and(trunc, ~term))
                    if len(cut):
                        final_obs = torch.as_tensor(np.stack([info["final_obs"][i] for i in cut]))
                        _, _, v_fin = model.act(final_obs)
                        b_final[t, cut] = v_fin
                obs = torch.as_tensor(next_obs)
            _, _, last_value = model.act(obs)
            adv, ret = compute_gae(b_rew, b_val, b_term, b_trunc, b_final, last_value, a["gamma"], a["gae_lambda"])
            o, ac = b_obs.reshape(T * N, *obs_shape), b_act.reshape(-1)
            adv, ret = adv.reshape(-1), ret.reshape(-1)
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            # Gradient trên cả đoạn thu: cộng dồn theo 16 phần để đỡ tốn bộ nhớ, rồi lấy trung bình.
            order = torch.randperm(T * N)
            parts = torch.tensor_split(order, a["num_minibatches"])
            sums = {"policy": 0.0, "total": 0.0}
            small = {"policy": [], "total": []}
            for idx in parts:
                pol, tot = losses(o[idx], ac[idx], adv[idx], ret[idx])
                gp = flat_grad(pol, retain=True)  # hai mất mát dùng chung một lần chạy xuôi
                gt = flat_grad(tot)
                small["policy"].append(float(gp.pow(2).sum()))
                small["total"].append(float(gt.pow(2).sum()))
                sums["policy"] = sums["policy"] + gp * len(idx)
                sums["total"] = sums["total"] + gt * len(idx)
            for k in ("policy", "total"):
                rec[k]["G"].append((sums[k] / (T * N)).numpy())
                rec[k]["g_small_sq"].append(float(np.mean(small[k])))
            print(f"rollout {r + 1}/{args.rollouts} done, {time.perf_counter() - t0:.0f}s", flush=True)
    finally:
        envs.close()

    b_small, b_big, R = T * N // a["num_minibatches"], T * N, args.rollouts
    out = {"ckpt": str(args.ckpt), "global_step": step, "levels": env_levels, "rollouts": R, "b_small": b_small,
           "b_big": b_big, "train_flag_rate": float(np.mean(flags)) if flags else None, "episodes": len(flags)}
    for k in ("policy", "total"):
        G = np.stack(rec[k]["G"])                       # (R, P)
        big_sq = (G ** 2).sum(axis=1)                   # |G_r|^2, cỡ b_big
        small_sq = np.array(rec[k]["g_small_sq"])       # trung bình |g_b|^2, cỡ b_small
        # Trong đoạn thu (McCandlish, A.1), trung bình tử và mẫu qua các đoạn rồi mới chia.
        g2 = (b_big * big_sq - b_small * small_sq) / (b_big - b_small)
        s = (small_sq - big_sq) / (1 / b_small - 1 / b_big)
        within = float(s.mean() / g2.mean())
        # Giữa các đoạn thu: phương sai của G_r quanh trung bình, |G|^2 không chệch = |mean|^2 - var/R.
        mean = G.mean(axis=0)
        var_r = float(((G - mean) ** 2).sum(axis=1).sum() / (R - 1))
        g2_between = float((mean ** 2).sum() - var_r / R)
        out[k] = {
            "B_simple_within_rollout": within,
            "G_sq_within": float(g2.mean()), "trace_sigma_within": float(s.mean()),
            "rollout_noise_to_signal": var_r / g2_between if g2_between > 0 else float("inf"),
            "B_simple_between_rollouts": b_big * var_r / g2_between if g2_between > 0 else float("inf"),
            "G_sq_between": g2_between, "rollout_var": var_r,
            "cos_between_rollouts": float(np.mean([G[i] @ G[j] / np.linalg.norm(G[i]) / np.linalg.norm(G[j])
                                                    for i in range(R) for j in range(i + 1, R)])),
        }
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
