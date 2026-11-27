"""Kiểm trước khi đổi γ (không cập nhật trọng số): khi mục tiêu của critic tính với γ khác, gradient của critic có lấn
gradient của actor trên thân CNN dùng chung không, và hai gradient cùng hay ngược hướng.

Tại θ của một checkpoint, chạy `--rollouts` đoạn thu num_envs x num_steps trên một màn (môi trường huấn luyện, có mẹo
huấn luyện). Với MỖI đoạn thu, tính GAE (λ của cấu hình) với từng giá trị trong `--gammas` trên CÙNG dữ liệu, rồi trên
CÙNG cách chia mini-batch (16 x 256, lợi thế chuẩn hóa theo mini-batch như lúc huấn luyện), tại tỉ số chính sách = 1:
  g_pi  gradient của mất mát chính sách -mean(Â log π) trên thân CNN (model.body)
  g_v   gradient của vf_coef * mean((V - R)^2) trên thân CNN
  g_e   gradient của -ent_coef * H trên thân CNN (ent_coef đúng như lịch ở bước của checkpoint)
Ghi cho từng mini-batch: |g_v| / |g_pi|, cos(g_v, g_pi), và hệ số mà phép cắt chuẩn gradient (max_grad_norm, trên MỌI
tham số) sẽ nhân vào bước đó; từ đó ra "độ lớn hiệu dụng của gradient actor sau khi cắt". Báo trung vị và khoảng
(phân vị 10–90) trên mọi mini-batch của mọi đoạn thu. Đây là chẩn đoán ở lúc bắt đầu học tiếp, không phải trong lúc học.

    python experiments/gamma_gradient_check.py --ckpt runs/8-levels/from-scratch-10m/checkpoints/step_006303744.pt \
        --level 2-1 --rollouts 6 --out runs/2-1/gamma-check/step_006303744.json
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
    p.add_argument("--level", required=True)
    p.add_argument("--rollouts", type=int, default=6)
    p.add_argument("--gammas", type=float, nargs="+", default=[0.9, 0.97])
    p.add_argument("--seed", type=int, default=4242)
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
    N, T, M = a["num_envs"], a["num_steps"], a["num_minibatches"]
    step = ck["global_step"]
    ent = a["ent_coef"]
    if a.get("ent_coef_final") is not None:
        ent = a["ent_coef"] + (a["ent_coef_final"] - a["ent_coef"]) * min(1.0, step / a["total_steps"])
    model = ActorCritic(len(ACTION_SETS[cfg["env"]["actions"]]), in_channels=cfg["env"]["frame_stack"],
                        **cfg.get("model", {}))
    model.load_state_dict(ck["model"])
    body = list(model.body.parameters())
    params = list(model.parameters())

    def grads(loss, wrt, retain):
        g = torch.autograd.grad(loss, wrt, retain_graph=retain, allow_unused=True)
        return torch.cat([(gi if gi is not None else torch.zeros_like(w)).reshape(-1) for gi, w in zip(g, wrt)])

    env_cfg = dict(cfg["env"], gamma=a["gamma"])
    envs = _make_vec_env(env_cfg, [args.level] * N, args.seed)
    obs_shape = envs.single_observation_space.shape
    b_obs = torch.zeros((T, N, *obs_shape), dtype=torch.uint8)
    b_act = torch.zeros((T, N), dtype=torch.long)
    b_val, b_rew, b_term, b_trunc, b_final = (torch.zeros((T, N)) for _ in range(5))
    rec = {g: {"ratio": [], "cos": [], "clip_scale": [], "actor_eff": [], "value_loss": [], "explained_var": []}
           for g in args.gammas}
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
                cut = np.flatnonzero(np.logical_and(trunc, ~term))
                if len(cut):
                    final_obs = torch.as_tensor(np.stack([info["final_obs"][i] for i in cut]))
                    _, _, v_fin = model.act(final_obs)
                    b_final[t, cut] = v_fin
                obs = torch.as_tensor(next_obs)
            _, _, last_value = model.act(obs)
            o, ac, v_old = b_obs.reshape(T * N, *obs_shape), b_act.reshape(-1), b_val.reshape(-1)
            parts = torch.tensor_split(torch.randperm(T * N), M)  # cùng cách chia cho mọi γ
            for g in args.gammas:
                adv, ret = compute_gae(b_rew, b_val, b_term, b_trunc, b_final, last_value, g, a["gae_lambda"])
                adv, ret = adv.reshape(-1), ret.reshape(-1)
                rv = ret.var(unbiased=False)
                rec[g]["explained_var"].append(float(1 - (ret - v_old).var(unbiased=False) / rv) if rv > 0 else 0.0)
                for idx in parts:
                    logits, values = model(o[idx])
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(ac[idx])
                    ah = adv[idx]
                    ah = (ah - ah.mean()) / (ah.std(unbiased=False) + 1e-8)
                    l_pi = -(torch.exp(logp - logp.detach()) * ah).mean()
                    l_v = a["vf_coef"] * ((values - ret[idx]) ** 2).mean()
                    l_e = -ent * dist.entropy().mean()
                    g_pi = grads(l_pi, body, True)
                    g_v = grads(l_v, body, True)
                    total = l_pi + l_v + l_e
                    g_all = grads(total, params, False)  # chuẩn mà clip_grad_norm_ thấy (mọi tham số)
                    n_pi, n_v = float(g_pi.norm()), float(g_v.norm())
                    scale = min(1.0, a["max_grad_norm"] / (float(g_all.norm()) + 1e-6))
                    rec[g]["ratio"].append(n_v / (n_pi + 1e-12))
                    rec[g]["cos"].append(float(g_pi @ g_v) / (n_pi * n_v + 1e-12))
                    rec[g]["clip_scale"].append(scale)
                    rec[g]["actor_eff"].append(scale * n_pi)
                    rec[g]["value_loss"].append(float(l_v) / a["vf_coef"])
            print(f"rollout {r + 1}/{args.rollouts}, {time.perf_counter() - t0:.0f}s", flush=True)
    finally:
        envs.close()

    def summ(xs):
        xs = np.asarray(xs, dtype=float)
        return {"median": float(np.median(xs)), "p10": float(np.percentile(xs, 10)), "p90": float(np.percentile(xs, 90))}

    out = {"ckpt": args.ckpt, "global_step": step, "level": args.level, "rollouts": args.rollouts,
           "ent_coef": ent, "vf_coef": a["vf_coef"], "max_grad_norm": a["max_grad_norm"], "gae_lambda": a["gae_lambda"],
           "by_gamma": {str(g): {k: summ(v) for k, v in rec[g].items()} for g in args.gammas}}
    base = str(args.gammas[0])
    for g in args.gammas[1:]:
        out[f"actor_eff_ratio_{g}_vs_{base}"] = (out["by_gamma"][str(g)]["actor_eff"]["median"]
                                                  / out["by_gamma"][base]["actor_eff"]["median"])
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
