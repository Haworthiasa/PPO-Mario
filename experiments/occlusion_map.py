"""Chẩn đoán (không huấn luyện): mạng dựa vào phần nào của màn hình? Che từng ô của ảnh đầu vào rồi đo mạng đổi ý
bao nhiêu (occlusion map, "bản đồ che ô").

    python experiments/occlusion_map.py --ckpt <ckpt> --trace-dir <thư mục của value_trace.py> --out-dir <thư mục>

Cần chạy experiments/value_trace.py trước: script đọc trace.npz (hành động của từng bước) và clusters.json (các cụm
nơi nhiều lượt kết thúc) trong --trace-dir. Với mỗi cụm nó chọn vài khung hình:
    hỏng     --k-before bước trước khi một lượt hỏng ở cụm kết thúc (không lấy lượt kẹt tới hết giờ)
    kẹt      --stuck-after bước sau lần cuối Mario còn tiến, ở các lượt đứng tới hết giờ
    đi qua   với mỗi khung trên, khung của một lượt đi qua được cụm đó, ở x gần nhất (ưu tiên độ cao gần nhau)
Khung hình lấy lại bằng cách chơi lại đúng lượt đó với các hành động đã ghi (trình giả lập tất định); script kiểm
rằng V tính lại trùng V trong trace.npz.

Che ô: một ô --patch x --patch điểm ảnh (mặc định 12x12 trên ảnh 84x84), trượt bước --stride, tô bằng độ sáng trung
bình của khung, che cùng chỗ ở cả 4 khung xếp chồng. Với mỗi vị trí đo
    dV  = V khi che - V gốc         (dương: che chỗ này làm critic lạc quan hơn, tức chỗ đó chứa thứ critic coi là xấu)
    KL  = KL(pi gốc || pi khi che)  (chính sách đổi bao nhiêu, đơn vị nat)
Bản đồ của mỗi điểm ảnh là trung bình các ô phủ lên điểm ảnh đó.

Để không chỉ "nhìn hình", script đếm phần khối lượng của |dV| và KL rơi vào bốn vùng: dải HUD (hàng 0..10), hộp quanh
Mario, hộp quanh các vật thể (địch, cây ăn thịt, lò xo: những thứ trò chơi giữ trong ô nhớ của địch), phần còn lại;
kèm diện tích của từng vùng. Vị trí Mario và vật thể đọc từ RAM của
trình giả lập CHỈ để gắn nhãn hình, không đưa vào mạng.

Tệp ghi ra trong --out-dir:
    frames.json            từng khung đã chọn: lượt, bước, x, V, các phần khối lượng theo vùng, dV trung bình theo vùng
    summary.txt            bảng tổng theo cụm và theo loại khung (hỏng / kẹt / đi qua)
    occlusion-x<lo>.png    mỗi cụm một hình: mỗi hàng một cặp (khung hỏng hoặc kẹt | khung đi qua ở cùng x)
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

HUD_ROWS = 11                       # hàng 0..10 của ảnh 84x84 là nền trời và chữ HUD (đo trong hud_mask_eval.py)
SCALE_X, SCALE_Y = 84 / 256, 84 / 240
MARIO_HALF, ENEMY_HALF = 9, 8       # nửa cạnh hộp quanh Mario / quanh vật thể, điểm ảnh của ảnh 84x84
# Địa chỉ RAM của Super Mario Bros (bản NES), chỉ dùng để gắn nhãn
PLAYER_REL_X, PLAYER_Y, PLAYER_Y_PAGE, PLAYER_SIZE = 0x03AD, 0x00CE, 0x00B5, 0x0754
ENEMY_FLAG, ENEMY_PAGE, ENEMY_X, ENEMY_Y, ENEMY_Y_PAGE = 0x000F, 0x006E, 0x0087, 0x00CF, 0x00B6
SCREEN_PAGE, SCREEN_X = 0x071A, 0x071C


def labels_from_ram(ram):
    """Tâm của Mario và các vật thể nằm trong ô nhớ "địch" (địch, cây ăn thịt, lò xo...) trên ảnh 84x84 (cột, hàng),
    hoặc None nếu ở ngoài màn hình. Độ lệch giữa tọa độ RAM và hình vẽ đo bằng mắt trên ảnh gốc 256x240 ở màn 2-1
    (Mario nhỏ được vẽ ở hàng y+16..y+32, Goomba ở y+8..y+24, lò xo ở y..y+24); hộp đủ rộng để bù sai lệch vài điểm ảnh."""
    mario = None
    if int(ram[PLAYER_Y_PAGE]) == 1:
        middle = 16 if int(ram[PLAYER_SIZE]) == 0 else 24       # 0 = Mario lớn (cao 32), 1 = Mario nhỏ (cao 16)
        mario = ((int(ram[PLAYER_REL_X]) + 8) * SCALE_X, (int(ram[PLAYER_Y]) + middle) * SCALE_Y)
    left = int(ram[SCREEN_PAGE]) * 256 + int(ram[SCREEN_X])
    enemies = []
    for i in range(5):
        if int(ram[ENEMY_FLAG + i]) == 0 or int(ram[ENEMY_Y_PAGE + i]) != 1:
            continue
        ex = int(ram[ENEMY_PAGE + i]) * 256 + int(ram[ENEMY_X + i]) - left
        if -8 <= ex < 256:
            enemies.append(((ex + 10) * SCALE_X, (int(ram[ENEMY_Y + i]) + 14) * SCALE_Y))
    return mario, enemies


def region_masks(mario, enemies):
    hud = np.zeros((84, 84), bool)
    hud[:HUD_ROWS] = True

    def box(center, half):
        m = np.zeros((84, 84), bool)
        if center is not None:
            c, r = center
            m[max(0, int(r - half)):max(0, int(r + half) + 1), max(0, int(c - half)):max(0, int(c + half) + 1)] = True
        return m

    mario_m = box(mario, MARIO_HALF) & ~hud
    enemy_m = np.zeros((84, 84), bool)
    for e in enemies:
        enemy_m |= box(e, ENEMY_HALF)
    enemy_m &= ~hud & ~mario_m
    rest = ~(hud | mario_m | enemy_m)
    return {"hud": hud, "mario": mario_m, "enemy": enemy_m, "rest": rest}


def occlude(model, obs, patch, stride):
    """Trả (V gốc, xác suất gốc, bản đồ dV 84x84, bản đồ KL 84x84, dV từng ô, KL từng ô)."""
    import torch
    positions = [(r, c) for r in range(0, 84 - patch + 1, stride) for c in range(0, 84 - patch + 1, stride)]
    means = obs.reshape(obs.shape[0], -1).mean(axis=1).round().astype(np.uint8)
    batch = np.repeat(obs[None], len(positions), axis=0)
    for i, (r, c) in enumerate(positions):
        batch[i, :, r:r + patch, c:c + patch] = means[:, None, None]
    with torch.no_grad():
        logits0, v0 = model(torch.as_tensor(obs[None]))
        logits, v = model(torch.as_tensor(batch))
    logp0 = torch.log_softmax(logits0, -1)
    kl = (logp0.exp() * (logp0 - torch.log_softmax(logits, -1))).sum(-1).numpy()
    dv = (v - v0).numpy()
    acc_dv, acc_kl, cnt = np.zeros((84, 84)), np.zeros((84, 84)), np.zeros((84, 84))
    for i, (r, c) in enumerate(positions):
        acc_dv[r:r + patch, c:c + patch] += dv[i]
        acc_kl[r:r + patch, c:c + patch] += kl[i]
        cnt[r:r + patch, c:c + patch] += 1
    return float(v0), logp0.exp()[0].numpy(), acc_dv / cnt, acc_kl / cnt, dv, kl


def choose_frames(tr, clusters, margin, k_before, stuck_after, per_cluster, escape_before=5):
    """Danh sách (cụm, loại, lượt, bước, lượt đi qua, bước đi qua)."""
    picks = []
    for c in clusters:
        passing = [e for e in range(tr.n_ep) if tr.cause[e] == "flag" or tr.x_end[e] > c["hi"] + margin]
        if not passing:
            continue
        wanted = []
        for e in c["dying"][:per_cluster]:
            if tr.n_steps[e] >= k_before:
                wanted.append(("hỏng", e, int(tr.n_steps[e]) - k_before))
        for e in c["stuck"][:per_cluster]:
            t = int(tr.last_progress[e]) + stuck_after
            if t < tr.n_steps[e]:
                wanted.append(("kẹt", e, t))
        used = set()
        # Ở chỗ kẹt, các lượt đi qua cũng đứng ở cùng x một lúc; khung đối chứng có ích hơn là lúc chúng vừa thoát
        # (--escape-before bước trước khi rời chỗ kẹt, thường là lúc Mario đã đứng trên lò xo).
        escaped = list(c.get("escapes") or [])
        for kind, e, t in wanted:
            if kind == "kẹt" and escaped:
                r = escaped.pop(0)
                picks.append({"cluster": c["lo"], "hi": c["hi"], "kind": kind, "ep": e, "t": t,
                              "pass_ep": r["episode"], "pass_t": max(0, r["leave"] - escape_before),
                              "pass_kind": "sắp thoát"})
                continue
            j = int(tr.start[e]) + t
            best = None
            for p in passing:
                s = tr.rows(p)
                cost = np.abs(tr.x[s].astype(int) - int(tr.x[j])) + 0.25 * np.abs(tr.y[s].astype(int) - int(tr.y[j]))
                cost = cost + (8 if p in used else 0)          # ưu tiên lấy các lượt đi qua khác nhau
                i = int(np.argmin(cost))
                if best is None or cost[i] < best[0]:
                    best = (cost[i], p, i)
            used.add(best[1])
            picks.append({"cluster": c["lo"], "hi": c["hi"], "kind": kind, "ep": e, "t": t,
                          "pass_ep": best[1], "pass_t": best[2], "pass_kind": "đi qua"})
    return picks


def grab(level, env_cfg, seed, tr, requests):
    """Chơi lại từng lượt bằng các hành động đã ghi; trả {(lượt, bước): (obs, nhãn từ RAM, x)}."""
    from mario_ppo.env import make_env
    from mario_ppo.evaluate import NOOP

    by_ep = defaultdict(set)
    for e, t in requests:
        by_ep[e].add(t)
    env = make_env(level, **dict(env_cfg, eval_mode=True))
    got = {}
    try:
        for e in sorted(by_ep):
            obs, info = env.reset(seed=seed + e)
            for _ in range(int(tr.noop_steps[e])):
                obs, _, term, trunc, info = env.step(NOOP)
            actions = tr.action[tr.rows(e)]
            last = max(by_ep[e])
            for t in range(last + 1):
                if t in by_ep[e]:
                    got[(e, t)] = (np.asarray(obs).copy(), labels_from_ram(env.unwrapped.ram), int(info.get("x_pos", 40)))
                if t < last:
                    obs, _, term, trunc, info = env.step(int(actions[t]))
    finally:
        env.close()
    return got


def draw(rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 6, figsize=(16.5, 2.85 * len(rows) + 0.6), squeeze=False)
    for ax_row, row in zip(axes, rows):
        lim_v = max(1e-3, max(np.abs(f["dv_map"]).max() for f in row))
        lim_k = max(1e-3, max(f["kl_map"].max() for f in row))
        for j, f in enumerate(row):
            a0, a1, a2 = ax_row[3 * j:3 * j + 3]
            frame = f["obs"][-1]
            for ax in (a0, a1, a2):
                ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([]); ax.set_yticks([])
            im1 = a1.imshow(f["dv_map"], cmap="RdBu_r", vmin=-lim_v, vmax=lim_v, alpha=0.6)
            im2 = a2.imshow(f["kl_map"], cmap="magma", vmin=0, vmax=lim_k, alpha=0.6)
            mario, enemies = f["labels"]
            for ax in (a0, a1, a2):
                if mario is not None:
                    ax.add_patch(plt.Rectangle((mario[0] - MARIO_HALF, mario[1] - MARIO_HALF), 2 * MARIO_HALF,
                                               2 * MARIO_HALF, fill=False, ec="#00e000", lw=0.9))
                for e in enemies:
                    ax.add_patch(plt.Rectangle((e[0] - ENEMY_HALF, e[1] - ENEMY_HALF), 2 * ENEMY_HALF, 2 * ENEMY_HALF,
                                               fill=False, ec="#ff2020", lw=0.9))
            a0.set_title(f"{f['kind']}: lượt {f['ep']}, bước {f['t']}\nx {f['x']}, V {f['v']:.2f}", fontsize=8)
            a1.set_title(f"dV khi che (đỏ: V tăng)\nmax |dV| {np.abs(f['dv_map']).max():.2f}", fontsize=8)
            a2.set_title(f"KL của chính sách\nmax {f['kl_map'].max():.2f} nat", fontsize=8)
            fig.colorbar(im1, ax=a1, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
            fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--k-before", type=int, default=4, help="khung hỏng lấy ở từng này bước trước khi lượt kết thúc")
    p.add_argument("--stuck-after", type=int, default=100, help="khung kẹt lấy ở từng này bước sau lần tiến cuối")
    p.add_argument("--per-cluster", type=int, default=3, help="số lượt hỏng (và số lượt kẹt) lấy ở mỗi cụm")
    p.add_argument("--escape-before", type=int, default=5,
                   help="ở cụm kẹt có lượt thoát được: khung đối chứng lấy từng này bước trước khi lượt đó rời chỗ kẹt")
    p.add_argument("--patch", type=int, default=12)
    p.add_argument("--stride", type=int, default=6)
    args = p.parse_args(argv)

    from experiments.value_trace import Trace
    from mario_ppo.evaluate import load_model

    trace_dir, out = Path(args.trace_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr = Trace(trace_dir / "trace.npz")
    info = json.loads((trace_dir / "clusters.json").read_text(encoding="utf-8"))
    picks = choose_frames(tr, info["clusters"], info["margin"], args.k_before, args.stuck_after, args.per_cluster,
                          args.escape_before)
    model, ck = load_model(args.ckpt)
    model.eval()
    level, seed = tr.meta["level"], tr.meta["seed"]
    got = grab(level, ck["config"]["env"], seed, tr, [(q["ep"], q["t"]) for q in picks] +
               [(q["pass_ep"], q["pass_t"]) for q in picks])

    records, by_cluster = [], defaultdict(list)
    worst_mismatch = 0.0
    for q in picks:
        row = []
        for kind, e, t in ((q["kind"], q["ep"], q["t"]), (q["pass_kind"], q["pass_ep"], q["pass_t"])):
            obs, labels, x = got[(e, t)]
            v0, probs, dv_map, kl_map, dv, kl = occlude(model, obs, args.patch, args.stride)
            worst_mismatch = max(worst_mismatch, abs(v0 - float(tr.v[int(tr.start[e]) + t])))
            masks = region_masks(*labels)
            mass_v, mass_k = np.abs(dv_map).sum(), kl_map.sum()
            rec = {"cluster": q["cluster"], "kind": kind, "ep": int(e), "t": int(t), "x": x, "v": v0,
                   "probs": [round(float(v), 4) for v in probs], "n_enemies": len(labels[1]),
                   "max_abs_dv": float(np.abs(dv).max()), "max_kl": float(kl.max()),
                   "area": {k: float(m.mean()) for k, m in masks.items()},
                   "share_abs_dv": {k: float(np.abs(dv_map)[m].sum() / mass_v) for k, m in masks.items()},
                   "share_kl": {k: float(kl_map[m].sum() / mass_k) for k, m in masks.items()},
                   "mean_dv": {k: (float(dv_map[m].mean()) if m.any() else None) for k, m in masks.items()}}
            records.append(rec)
            row.append(dict(rec, obs=obs, labels=labels, dv_map=dv_map, kl_map=kl_map))
        by_cluster[(q["cluster"], q["hi"])].append(row)

    for (lo, hi), rows in by_cluster.items():
        draw(rows, out / f"occlusion-x{lo}.png",
             f"Màn {level}, cụm x {lo}-{hi}: che ô {args.patch}x{args.patch} bước {args.stride} ở cả 4 khung. "
             f"Khung xanh lá = Mario, khung đỏ = địch/cây ăn thịt/lò xo (đọc từ RAM, chỉ để gắn nhãn).\n"
             f"Mỗi hàng: trái là lượt hỏng/kẹt, "
             f"phải là lượt đi qua ở x gần nhất (ở chỗ kẹt: lượt sắp thoát, {args.escape_before} bước trước khi rời chỗ kẹt)")
    (out / "frames.json").write_text(json.dumps({"checkpoint": args.ckpt, "level": level, "patch": args.patch,
                                                 "stride": args.stride, "frames": records}, indent=1,
                                                ensure_ascii=False), encoding="utf-8")

    lines = [f"=== che ô trên {level}, {args.ckpt}: {len(records)} khung, ô {args.patch}x{args.patch}, bước {args.stride} ===",
             f"V tính lại khớp trace.npz: sai lệch lớn nhất {worst_mismatch:.2e}",
             "phần khối lượng |dV| (và KL) rơi vào từng vùng; diện tích = phần ảnh mà vùng đó chiếm (trung bình các khung)",
             "vật thể = địch, cây ăn thịt, lò xo; dV TB vật thể = dV trung bình trên vùng hộp vật thể (dương: che nó làm V tăng)"]
    groups = defaultdict(list)
    for r in records:
        groups[(r["cluster"], r["kind"])].append(r)
        groups[("tất cả", r["kind"])].append(r)
    lines.append("   cụm    loại     khung  |dV|: HUD  Mario  vật thể còn lại |  KL: HUD  Mario  vật thể còn lại |"
                 "  diện tích: HUD  Mario  vật thể | max|dV|  maxKL  dV TB vật thể  dV TB Mario")
    for (lo, kind), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        sv = {k: np.mean([r["share_abs_dv"][k] for r in rs]) for k in ("hud", "mario", "enemy", "rest")}
        sk = {k: np.mean([r["share_kl"][k] for r in rs]) for k in ("hud", "mario", "enemy", "rest")}
        ar = {k: np.mean([r["area"][k] for r in rs]) for k in ("hud", "mario", "enemy")}
        de = [r["mean_dv"]["enemy"] for r in rs if r["mean_dv"]["enemy"] is not None]
        dm = [r["mean_dv"]["mario"] for r in rs if r["mean_dv"]["mario"] is not None]
        lines.append(f"  {str(lo):>6}  {kind:7s}  {len(rs):4d}   " +
                     "  ".join(f"{sv[k]:5.0%}" for k in ("hud", "mario", "enemy", "rest")) + "  |      " +
                     "  ".join(f"{sk[k]:5.0%}" for k in ("hud", "mario", "enemy", "rest")) + "  |          " +
                     "  ".join(f"{ar[k]:5.0%}" for k in ("hud", "mario", "enemy")) +
                     f" |  {np.mean([r['max_abs_dv'] for r in rs]):5.2f}  {np.mean([r['max_kl'] for r in rs]):5.2f}"
                     f"   {(np.mean(de) if de else float('nan')):+6.2f} ({len(de)})  {(np.mean(dm) if dm else float('nan')):+6.2f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
