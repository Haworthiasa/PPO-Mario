"""Đo trước khi huấn luyện (không cập nhật trọng số): với CÙNG dữ liệu, các cách ước lượng lợi thế (γ, λ, hệ số thưởng)
cho gradient chính sách nhiễu tới mức nào.

Vì sao: `docs/config-math.md`, Mục 2 đo được gradient chính sách của một đoạn thu 4.096 mẫu chỉ có 2–19% năng lượng là
tín hiệu (B_simple của actor 17 nghìn tới hơn 100 nghìn mẫu). Khi B ≪ B_simple, số mẫu cần để học tỉ lệ thuận với
B_simple (McCandlish và cộng sự, 2018). Nên ước lượng lợi thế nào có B_simple nhỏ hơn thì, nếu không chệch nhiều,
học tốn ít bước hơn. λ < 1 giảm phương sai bằng cách dựa vào critic; γ lớn hơn nhìn xa hơn nhưng cộng thêm phương sai.

Cách làm: tại θ của một checkpoint, chạy `--rollouts` đoạn thu num_envs x num_steps trên môi trường huấn luyện (có
mẹo huấn luyện). Với mỗi đoạn thu và mỗi ước lượng (γ, λ, s) trong `--estimators`, tính GAE trên phần thưởng nhân s
(giá trị của critic giữ nguyên), chuẩn hóa lợi thế trên cả đoạn, rồi tính gradient của -mean(Â log π) trên mọi tham
số, ở 16 mini-batch 256 mẫu (cùng cách chia cho mọi ước lượng) và trên cả đoạn. Từ đó ra B_simple trong đoạn và giữa
các đoạn như `experiments/gradient_noise.py`, cộng cos giữa gradient trung bình của các ước lượng với ước lượng đầu tiên.

Giới hạn: critic của checkpoint học cho γ và hệ số thưởng của lượt gốc. Với γ khác, V chỉ đúng bậc độ lớn khi
s = (1 - γ) / (1 - γ_gốc) (phần thưởng đều thì V ≈ s·r̄ / (1 - γ)), nên số đo của γ khác kèm λ < 1 chỉ là xấp xỉ.

    python experiments/estimator_noise.py --ckpt runs/1-1/run-1/checkpoints/step_000401408.pt --rollouts 8 \
        --estimators 0.9,1.0,1 0.9,0.95,1 0.97,1.0,0.3 0.97,0.95,0.3 --out <tệp.json>
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
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--estimators", nargs="+", default=["0.9,1.0,1", "0.9,0.95,1", "0.97,1.0,0.3", "0.97,0.95,0.3"],
                   help="mỗi mục là γ,λ,hệ số thưởng")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import torch
    from mario_ppo.model import ActorCritic
    from mario_ppo.ppo import compute_gae
    from mario_ppo.train_joint import _make_vec_env
    from mario_ppo.wrappers import ACTION_SETS

    torch.set_num_threads(6)
    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    a = cfg["algo"]
    level = (cfg.get("levels") or [cfg["level"]])[0]
    N, T, M = a["num_envs"], a["num_steps"], a["num_minibatches"]
    ests = [tuple(float(v) for v in e.split(",")) for e in args.estimators]
    names = [f"γ {g:g}, λ {lam:g}, thưởng x{s:g}" for g, lam, s in ests]

    model = ActorCritic(len(ACTION_SETS[cfg["env"]["actions"]]), in_channels=cfg["env"]["frame_stack"],
                        **cfg.get("model", {}))
    model.load_state_dict(ck["model"])
    params = list(model.parameters())

    envs = _make_vec_env(dict(cfg["env"], gamma=a["gamma"]), [level] * N, args.seed)
    obs_shape = envs.single_observation_space.shape
    b_obs = torch.zeros((T, N, *obs_shape), dtype=torch.uint8)
    b_act = torch.zeros((T, N), dtype=torch.long)
    b_val, b_rew, b_term, b_trunc, b_final = (torch.zeros((T, N)) for _ in range(5))
    G = {k: [] for k in range(len(ests))}       # gradient trên cả đoạn, mỗi đoạn một véc-tơ
    small_sq = {k: [] for k in range(len(ests))}  # trung bình |g_b|^2 trên 16 mini-batch
    ev = {k: [] for k in range(len(ests))}        # phần phương sai của mục tiêu mà V giải thích (sau khi nhân s)
    adv_std = {k: [] for k in range(len(ests))}   # độ lệch chuẩn của lợi thế thô, chia cho s
    flags, t0 = [], time.perf_counter()
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
                        _, _, v_fin = model.act(torch.as_tensor(np.stack([info["final_obs"][i] for i in cut])))
                        b_final[t, cut] = v_fin
                obs = torch.as_tensor(next_obs)
            _, _, last_value = model.act(obs)
            advs = []
            for k, (g, lam, s) in enumerate(ests):
                adv, ret = compute_gae(b_rew * s, b_val, b_term, b_trunc, b_final, last_value, g, lam)
                adv, ret = adv.reshape(-1), ret.reshape(-1)
                ev[k].append(float(1 - (ret - b_val.reshape(-1)).var() / ret.var()))
                adv_std[k].append(float(adv.std() / s))
                advs.append((adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8))
            o, ac = b_obs.reshape(T * N, *obs_shape), b_act.reshape(-1)
            order = torch.randperm(T * N)
            sums = [0.0] * len(ests)
            sq = [[] for _ in ests]
            for idx in torch.tensor_split(order, M):
                logits, _ = model(o[idx])
                logp = torch.distributions.Categorical(logits=logits).log_prob(ac[idx])
                for k in range(len(ests)):
                    grads = torch.autograd.grad(-(advs[k][idx] * logp).mean(), params, retain_graph=True,
                                                allow_unused=True)
                    gk = torch.cat([(q if q is not None else torch.zeros_like(w)).reshape(-1)
                                    for q, w in zip(grads, params)])
                    sq[k].append(float(gk.pow(2).sum()))
                    sums[k] = sums[k] + gk * len(idx)
            for k in range(len(ests)):
                G[k].append((sums[k] / (T * N)).numpy())
                small_sq[k].append(float(np.mean(sq[k])))
            print(f"rollout {r + 1}/{args.rollouts} done, {time.perf_counter() - t0:.0f}s", flush=True)
    finally:
        envs.close()

    b_small, b_big, R = T * N // M, T * N, args.rollouts
    out = {"ckpt": str(args.ckpt), "global_step": ck["global_step"], "level": level, "rollouts": R,
           "train_flag_rate": float(np.mean(flags)) if flags else None, "episodes": len(flags), "estimators": {}}
    mean0 = None
    for k, name in enumerate(names):
        Gk = np.stack(G[k])
        big_sq = (Gk ** 2).sum(axis=1)
        ss = np.array(small_sq[k])
        g2 = (b_big * big_sq - b_small * ss) / (b_big - b_small)
        tr = (ss - big_sq) / (1 / b_small - 1 / b_big)
        mean = Gk.mean(axis=0)
        var_r = float(((Gk - mean) ** 2).sum(axis=1).sum() / (R - 1))
        g2_between = float((mean ** 2).sum() - var_r / R)
        if mean0 is None:
            mean0 = mean
        out["estimators"][name] = {
            "B_simple_within": float(tr.mean() / g2.mean()) if g2.mean() > 0 else float("inf"),
            "B_simple_between": b_big * var_r / g2_between if g2_between > 0 else float("inf"),
            "signal_fraction_one_rollout": g2_between / (g2_between + var_r) if g2_between > 0 else 0.0,
            "cos_with_first": float(mean @ mean0 / np.linalg.norm(mean) / np.linalg.norm(mean0)),
            "explained_var_of_targets": float(np.mean(ev[k])), "raw_adv_std_over_s": float(np.mean(adv_std[k])),
        }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
