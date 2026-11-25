"""Số liệu cho docs/config-review.md: chẩn đoán PPO, thời gian mỗi lần cập nhật, phân bổ bước, xác suất qua ngưỡng,
phép tính γ và chi phí CPU của hai kiến trúc mạng. Chỉ đọc runs/, không huấn luyện.

    python experiments/config_review.py            # mọi phần trừ đo tốc độ mạng
    python experiments/config_review.py --bench    # thêm đo một bước gradient của Nature-CNN và IMPALA-CNN (~1 phút)
"""
import argparse
import csv
import sys
import time
from math import comb
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
DIAG_RUNS = ["1-1/run-1", "4-1/from-scratch", "1-1+4-1/from-scratch", "2-3/from-scratch-3m",
             "8-levels/from-scratch-10m", "2-1/wall-cut-200", "2-1/wall-cut-2000"]
COLS = ["lr", "approx_kl", "clipfrac", "entropy", "explained_var", "grad_norm", "value_loss",
        "minibatch_updates", "ent_coef", "flag_rate"]
FULL_UPDATE = 160  # 10 epoch x 16 mini-batch: lần cập nhật không bị target_kl dừng sớm


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
        except ValueError:
            pass
    return out


def diagnostics():
    print("== 1. Chẩn đoán PPO theo năm đoạn bằng nhau của mỗi lượt (trung bình các lần cập nhật)")
    for run in DIAG_RUNS:
        d = load(RUNS / run / "progress.csv")
        n = len(d["global_step"])
        mb = d["minibatch_updates"]
        print(f"\n  {run}: {n} lần cập nhật; dừng sớm vì target_kl {(mb < FULL_UPDATE).sum()}/{n} "
              f"({100 * (mb < FULL_UPDATE).mean():.1f}%); số epoch chạy trung bình {mb.mean() / 16:.2f}/10")
        cols = [c for c in COLS if c in d]
        print("  đoạn (bước)      " + " ".join(f"{c[:10]:>10}" for c in cols))
        edges = np.linspace(0, n, 6).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            vals = " ".join(f"{np.nanmean(d[c][a:b]):10.4g}" for c in cols)
            print(f"  {d['global_step'][a] / 1e6:5.2f}M-{d['global_step'][b - 1] / 1e6:5.2f}M {vals}")
        print(f"  grad_norm trước khi cắt: trung vị {np.median(d['grad_norm']):.2f} (trần max_grad_norm 0,5); "
              f"approx_kl trung vị {np.median(d['approx_kl']):.4f}, p90 {np.percentile(d['approx_kl'], 90):.4f}; "
              f"explained_var trung vị {np.median(d['explained_var']):.2f}")


def update_time():
    print("\n== 2. Thời gian mỗi lần cập nhật = a (thu dữ liệu + phần khác) + b x số bước gradient")
    print("   (suy từ cột sps, vốn là trung bình cộng dồn từ đầu mỗi đoạn chạy)")
    for run in ["8-levels/from-scratch-10m", "1-1+4-1/from-scratch", "2-3/from-scratch-3m"]:
        d = load(RUNS / run / "progress.csv")
        step, sps, mb = d["global_step"], d["sps"], d["minibatch_updates"]
        dts, mbs = [], []
        seg_start, prev = step[0] - 4096, None
        for i in range(len(step)):
            if i > 0 and (step[i] - step[i - 1] != 4096 or abs(sps[i] - sps[i - 1]) > 15):  # chỗ nối resume=
                seg_start = step[i - 1]
                prev = 0.0 if step[i] - step[i - 1] == 4096 else None
            elapsed = (step[i] - seg_start) / sps[i]
            if prev is not None and i > 0 and 0 < elapsed - prev < 200:
                dts.append(elapsed - prev)
                mbs.append(mb[i])
            prev = elapsed
        dts, mbs = np.array(dts), np.array(mbs)
        keep = (dts > np.percentile(dts, 2)) & (dts < np.percentile(dts, 98))
        (a, b), *_ = np.linalg.lstsq(np.vstack([np.ones(keep.sum()), mbs[keep]]).T, dts[keep], rcond=None)
        full = a + b * FULL_UPDATE
        print(f"  {run}: a = {a:.1f} s, b = {b * 1000:.0f} ms; 10 epoch đủ = {b * FULL_UPDATE:.1f} s "
              f"trên {full:.1f} s mỗi lần cập nhật ({100 * b * FULL_UPDATE / full:.0f}%)")


def allocation():
    print("\n== 3. Lượt 8 màn: số bước mỗi màn và kết quả cuối (200 lượt, seed 7 + seed 11)")
    a = load(RUNS / "8-levels/from-scratch-10m/allocation.csv")
    final = {"1-1": 97.5, "2-1": 3.5, "3-1": 98.5, "3-2": 97.0, "4-1": 99.5, "5-1": 89.5, "5-2": 75.0, "6-1": 99.0}
    phase1 = 2002944 / 8  # đoạn 0 -> 2.002.944: mỗi màn một môi trường
    total = sum(a[f"steps_{lv}"].sum() + phase1 for lv in final)
    for lv, flag in final.items():
        s = a[f"steps_{lv}"].sum() + phase1
        print(f"  {lv}: {s / 1e6:4.2f}M bước ({100 * s / total:4.1f}% tổng) -> {flag:5.1f}% chạm cờ")


def p_pass(p, n=20, k=16):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def pass_probability():
    print("\n== 4. Xác suất đạt >= 16/20 ở một lần chấm mới nếu tỉ lệ thật là p")
    for name, p in [("2-3 nộp (173/200)", 173 / 200), ("1-1 nộp (89/100)", 0.89), ("4-1 nộp (96/100)", 0.96),
                    ("p = 0,80", 0.80), ("p = 0,85", 0.85), ("p = 0,90", 0.90)]:
        expected = sum(comb(20, i) * p ** i * (1 - p) ** (20 - i) * min(1, i / 20 / 0.8) for i in range(21))
        print(f"  {name:18s}: P(>= 16/20) = {p_pass(p):.3f}; kỳ vọng phần điểm min(1, f/0,8) = {expected:.3f}")


def discount():
    print("\n== 5. Hệ số chiết khấu (1 bước tác tử = 4 khung = 1/15 giây)")
    r_run = 10 / 30  # chạy ~2,5 điểm ảnh/khung x 4 khung, nhân hệ số 5/150 của môi trường
    for g in (0.9, 0.95, 0.97, 0.99):
        prize = [g ** dly * r_run / (1 - g) for dly in (5, 10, 15, 20)]
        print(f"  γ = {g}: tầm 1/(1-γ) = {1 / (1 - g):5.1f} bước ({1 / (1 - g) / 15:4.2f} s), nửa đời "
              f"{np.log(0.5) / np.log(g):5.1f} bước; trọng số ở 10/30/60 bước {g ** 10:.3f}/{g ** 30:.3f}/"
              f"{g ** 60:.4f}; V khi chạy đều ~ {r_run / (1 - g):5.1f}; giá trị 'đã qua chướng ngại' nhìn từ "
              f"5/10/15/20 bước trước: " + "/".join(f"{v:.2f}" for v in prize))


def bench():
    import torch
    from torch import nn
    sys.path.insert(0, str(REPO))
    from mario_ppo.model import ActorCritic

    torch.set_num_threads(6)

    class Res(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c1, self.c2 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)

        def forward(self, x):
            return x + self.c2(torch.relu(self.c1(torch.relu(x))))

    class Impala(nn.Module):  # IMPALA-CNN 16-32-32 (Espeholt và cộng sự 2018), đầu ra như ActorCritic
        def __init__(self, n_actions=7):
            super().__init__()
            layers, c_in = [], 4
            for c in (16, 32, 32):
                layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.MaxPool2d(3, 2, padding=1), Res(c), Res(c)]
                c_in = c
            self.body = nn.Sequential(*layers, nn.ReLU(), nn.Flatten())
            with torch.no_grad():
                n = self.body(torch.zeros(1, 4, 84, 84)).shape[1]
            self.fc, self.pi, self.v = nn.Linear(n, 256), nn.Linear(256, n_actions), nn.Linear(256, 1)

        def forward(self, obs):
            h = torch.relu(self.fc(self.body(obs.float() / 255.0)))
            return self.pi(h), self.v(h).squeeze(-1)

    def macs(model):
        total = 0

        def hook(m, i, o):
            nonlocal total
            if isinstance(m, nn.Conv2d):
                total += o.numel() * m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            else:
                total += m.in_features * m.out_features
        hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        with torch.no_grad():
            model(torch.zeros(1, 4, 84, 84, dtype=torch.uint8))
        for h in hs:
            h.remove()
        return total

    def step_time(model, batch=256, reps=15):
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        x = torch.randint(0, 255, (batch, 4, 84, 84), dtype=torch.uint8)
        for i in range(reps + 3):
            if i == 3:
                t = time.perf_counter()
            logits, v = model(x)
            loss = logits.logsumexp(-1).mean() + v.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return (time.perf_counter() - t) / reps

    print("\n== 6. Chi phí một bước gradient (mini-batch 256) trên CPU, 6 luồng torch")
    for name, m in [("Nature-CNN (model.py)", ActorCritic(7)), ("IMPALA-CNN 16-32-32", Impala())]:
        print(f"  {name}: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M tham số, "
              f"{macs(m) / 1e6:.1f}M phép nhân-cộng mỗi ảnh, {step_time(m) * 1000:.0f} ms mỗi bước")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", action="store_true", help="đo thêm tốc độ hai kiến trúc mạng")
    args = p.parse_args()
    diagnostics()
    update_time()
    allocation()
    pass_probability()
    discount()
    if args.bench:
        bench()
