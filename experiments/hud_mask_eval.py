"""Chẩn đoán (không huấn luyện): chấm một checkpoint theo đúng giao thức của evaluate.py, nhưng sửa dòng chữ HUD
(điểm, xu, tên màn, đồng hồ) trên khung hình trước khi đưa vào mạng. Dùng để xem chính sách có dựa vào HUD không.

    python experiments/hud_mask_eval.py --ckpt <checkpoint.pt> --level 1-1 --mode frozen --other-level 4-1 \
        --episodes 100 --seed 7 --out <tệp.json>

Trong khung 84x84 của nhóm, chữ HUD nằm ở hàng 5..10 (hàng 0..4 là nền, hàng 11 trở đi là sân chơi); đã đo ở 1-1,
4-1 và 3-2. Bốn cách sửa:
  none    không sửa gì (đối chứng: phải trùng kết quả của evaluate.py với cùng seed)
  blank   xóa chữ, tô bằng màu nền của chính khung đó (lấy từ hàng 0)
  frozen  dán HUD của khung đầu lượt cùng màn (điểm 0, đồng hồ đầy, tên màn đúng): hình quen nhưng không còn đồng hồ
  swap    dán HUD đầu lượt của màn khác (--other-level): như frozen nhưng tên màn sai

Vòng chơi là run_episodes của evaluate.py (không sửa tệp đó); script chỉ bọc model.act. Không dùng để chấm điểm.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from mario_ppo.env import make_env
from mario_ppo.evaluate import load_model, run_episodes

HUD_ROWS = slice(5, 11)


def reference_hud(level, env_cfg):
    """Các hàng HUD của quan sát đầu lượt (khung mới nhất), dạng tensor uint8 (6, 84)."""
    env = make_env(level, **dict(env_cfg, eval_mode=True))
    try:
        obs, _ = env.reset(seed=0)
    finally:
        env.close()
    return torch.as_tensor(np.asarray(obs)[-1][HUD_ROWS].copy())


class HudEdited(torch.nn.Module):
    """Bọc mạng: sửa các hàng HUD của mọi khung trong chồng 4 khung rồi gọi mạng gốc."""

    def __init__(self, model, mode, reference=None):
        super().__init__()
        self.model, self.mode, self.reference = model, mode, reference

    def edit(self, obs):
        if self.mode == "none":
            return obs
        obs = obs.clone()
        if self.mode == "blank":
            obs[..., HUD_ROWS, :] = obs[..., 0:1, :]  # hàng 0 là nền của chính khung đó
        else:  # frozen, swap
            obs[..., HUD_ROWS, :] = self.reference
        return obs

    def act(self, obs):
        return self.model.act(self.edit(obs))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--level", required=True)
    p.add_argument("--mode", choices=("none", "blank", "frozen", "swap"), required=True)
    p.add_argument("--other-level", default=None, help="màn cho HUD của cách swap")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    torch.set_num_threads(1)  # nhiều lần chấm chạy song song, mỗi lần một luồng
    model, ck = load_model(args.ckpt)
    env_cfg = ck["config"]["env"]
    reference = None
    if args.mode == "frozen":
        reference = reference_hud(args.level, env_cfg)
    elif args.mode == "swap":
        if not args.other_level:
            p.error("--mode swap cần --other-level")
        reference = reference_hud(args.other_level, env_cfg)
    res = run_episodes(HudEdited(model, args.mode, reference), args.level, env_cfg, episodes=args.episodes,
                       seed=args.seed, max_steps=args.max_steps)
    res = {"level": args.level, "checkpoint": str(args.ckpt), "global_step": int(ck.get("global_step", 0)),
           "seed": args.seed, "hud_mode": args.mode, "hud_from": args.other_level if args.mode == "swap" else None,
           "flags": round(res["flag_rate"] * res["episodes"]), **res,
           "mean_x_pos": float(np.mean(res["x_pos"]))}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"{args.level} hud={args.mode}: {res['flags']}/{res['episodes']} cờ, x trung bình {res['mean_x_pos']:.0f}")


if __name__ == "__main__":
    main()
