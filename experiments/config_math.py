"""Số liệu và hình cho docs/config-math.md: phân tích toán học của cấu hình từ các lượt đã chạy. Chỉ đọc runs/.

    python experiments/config_math.py                 # in số liệu, ghi hình vào docs/config-math/

Các phần:
  A. Vùng tin cậy: approxKL theo tốc độ học và entropy; tốc độ học "cần" để đạt một mức KL mục tiêu.
  B. Đường cong học: số bước riêng của màn tới 50% / 80% chạm cờ; khớp đường logistic vào các lần chấm 100 lượt;
     độ dốc cuối lượt của từng màn trong lượt 8 màn (điều kiện chia bước tối ưu).
  C. Thống kê: độ phân tán giữa các checkpoint so với nhị thức; số lượt cần để phát hiện một khác biệt.
  D. Entropy: cân bằng của phần thưởng entropy; mô hình "vài quyết định then chốt" cho khoảng cách lấy mẫu / tham lam.
  E. Hình hệ số chiết khấu.
  F. Thang nhiễu gradient (đọc kết quả của experiments/gradient_noise.py nếu đã có).
"""
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import optimize, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedFormatter, FuncFormatter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
FIG = REPO / "docs" / "config-math"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LEVELS8 = ["1-1", "2-1", "3-1", "3-2", "4-1", "5-1", "5-2", "6-1"]
KL_TARGET = 0.02


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


def style(ax, title, note=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=18 if note else 6)
    if note:
        ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))  # dấu thập phân kiểu Việt, như mario_ppo/plot.py
    # Không đè trục đã có nhãn chữ (tên màn, tên checkpoint) hay trục log.
    if not isinstance(ax.xaxis.get_major_formatter(), FixedFormatter):
        ax.xaxis.set_major_formatter(comma)
    if ax.get_yscale() == "linear" and not isinstance(ax.yaxis.get_major_formatter(), FixedFormatter):
        ax.yaxis.set_major_formatter(comma)


def new_fig(ncols, w=11.5, h=4.2):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE)
    return fig, np.atleast_1d(axes)


def rolling_median(x, k=15):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - k // 2): i + k // 2 + 1]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = np.median(w)
    return out


# ------------------------------------------------------------------------------------------------------------
# A. Vùng tin cậy
# ------------------------------------------------------------------------------------------------------------
TR_RUNS = [("1-1 (một màn)", "1-1/run-1"), ("4-1 (một màn)", "4-1/from-scratch"),
           ("2-3, 3 triệu", "2-3/from-scratch-3m"), ("1-1 + 4-1", "1-1+4-1/from-scratch"),
           ("8 màn", "8-levels/from-scratch-10m")]


def trust_region():
    print("== A. Vùng tin cậy: KL theo tốc độ học (giả định KL ≈ ½ Δθᵀ F Δθ, Δθ tỉ lệ với tốc độ học)")
    fig, (ax1, ax2) = new_fig(2)
    for i, (name, run) in enumerate(TR_RUNS):
        d = load(RUNS / run / "progress.csv")
        lr, kl, ent, mb = d["lr"], d["approx_kl"], d["entropy"], d["minibatch_updates"]
        full = mb >= 160
        frac = (d["global_step"] - d["global_step"][0]) / (d["global_step"][-1] - d["global_step"][0])
        with np.errstate(divide="ignore", invalid="ignore"):
            need = lr * np.sqrt(KL_TARGET / kl)    # tốc độ học cho KL = mục tiêu, cùng độ nhạy
        ratio = np.where(full & (kl > 0), need / lr, np.nan)  # dừng sớm thì KL đã chạm trần: bỏ
        sens = kl / lr ** 2
        q = np.linspace(0, len(lr), 6).astype(int)
        parts = []
        for a, b in zip(q[:-1], q[1:]):
            r = ratio[a:b]
            parts.append(f"{np.nanmedian(r):.2f}" if np.isfinite(r).any() else "  - ")
        fit_mask = full & (lr > 0) & (kl > 0) & np.isfinite(sens)
        slope = np.polyfit(ent[fit_mask], np.log(sens[fit_mask]), 1)[0] if fit_mask.sum() > 20 else np.nan
        span = (np.nanpercentile(ent, 95), np.nanpercentile(ent, 5))
        fold = np.exp(slope * (span[1] - span[0])) if np.isfinite(slope) else np.nan
        print(f"  {name:14s}: lr cần / lr thật theo 5 đoạn = {' '.join(parts)}; "
              f"KL/lr² tăng {fold:.0f} lần khi entropy giảm {span[0]:.2f} -> {span[1]:.2f}")
        ax1.plot(frac, rolling_median(ratio, 21), color=SERIES[i], lw=2, label=name)
        ax2.scatter(ent[fit_mask], sens[fit_mask], s=6, color=SERIES[i], alpha=0.35, linewidths=0)
    ax1.axhline(1, color=MUTED, lw=1, ls="--")
    ax1.text(0.5, 0.93, "đúng mức KL 0,02", color=MUTED, fontsize=8, ha="center", va="top")
    ax1.set_ylim(0, 3.5)
    ax1.set_xlabel("phần của lượt chạy (tốc độ học giảm tuyến tính về 0 ở cuối)", fontsize=8, color=MUTED)
    style(ax1, "Tốc độ học cần cho KL = 0,02, chia cho tốc độ học thật",
          "> 1: bước cập nhật ngắn hơn mức cho phép; chỉ tính lần cập nhật không bị dừng sớm")
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", ncol=2)
    ax2.set_yscale("log")
    ax2.invert_xaxis()
    ax2.set_xlabel("entropy của chính sách (giảm dần theo thời gian →)", fontsize=8, color=MUTED)
    style(ax2, "Độ nhạy KL / lr² theo entropy", "mỗi chấm là một lần cập nhật; màu như hình trái")
    fig.tight_layout()
    fig.savefig(FIG / "trust-region.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------
# B. Đường cong học
# ------------------------------------------------------------------------------------------------------------
def level_steps_8():
    """Số bước riêng của từng màn tại mỗi lần cập nhật của lượt 8 màn."""
    d = load(RUNS / "8-levels/from-scratch-10m/progress.csv")
    a = load(RUNS / "8-levels/from-scratch-10m/allocation.csv")
    g = d["global_step"]
    out = {}
    for lv in LEVELS8:
        s = np.minimum(g, 2002944) / 8
        cum = np.concatenate([[0.0], np.cumsum(a[f"steps_{lv}"])])
        # dòng thứ j của allocation.csv ứng với lần cập nhật kết thúc ở a["global_step"][j]
        idx = np.searchsorted(a["global_step"], g, side="right")
        s = s + cum[np.clip(idx, 0, len(cum) - 1)]
        out[lv] = s
    return d, out


def first_cross(steps, rate, thr, episodes):
    # Chỉ tính khi cửa sổ đã đủ 100 lượt: sau mỗi lần resume= cửa sổ bắt đầu lại từ rỗng, vài lượt đầu làm tỉ lệ nhảy.
    hit = np.flatnonzero((rate >= thr) & (episodes >= 100))
    return steps[hit[0]] if len(hit) else np.nan


def logistic_fit(steps, k, n):
    """Khớp p(s) = A / (1 + exp(-(s - s0) / w)) bằng hợp lý cực đại nhị thức."""
    s = steps / 1e6

    def nll(th):
        A, s0, w = 1 / (1 + np.exp(-th[0])), th[1], np.exp(th[2])
        p = np.clip(A / (1 + np.exp(-(s - s0) / w)), 1e-6, 1 - 1e-6)
        return -(k * np.log(p) + (n - k) * np.log(1 - p)).sum()
    best = None
    for s0 in np.linspace(s.min(), s.max(), 7):
        for lw in (-3, -2, -1):
            r = optimize.minimize(nll, [3.0, s0, lw], method="Nelder-Mead", options={"maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
    A, s0, w = 1 / (1 + np.exp(-best.x[0])), best.x[1], np.exp(best.x[2])
    return A, s0, w


def eval_series(folder):
    steps, k = [], []
    for f in sorted(Path(folder).glob("100-seed7_*.json")):
        r = json.loads(f.read_text())
        steps.append(r["global_step"])
        k.append(round(r["flag_rate"] * r["episodes"]))
    return np.array(steps, float), np.array(k, float)


def learning_curves():
    print("\n== B1. Số bước RIÊNG của màn tới khi tỉ lệ chạm cờ lúc huấn luyện (100 lượt gần nhất) đạt 50% / 80%")
    rows = []
    d = load(RUNS / "1-1/run-1/progress.csv")
    rows.append(("1-1", "một màn (lr 1,5e-4, lịch 2M)", d["global_step"], d["flag_rate"], d["episodes"]))
    d = load(RUNS / "4-1/from-scratch/progress.csv")
    rows.append(("4-1", "một màn (lịch 1,2M)", d["global_step"], d["flag_rate"], d["episodes"]))
    d = load(RUNS / "1-1+4-1/from-scratch/progress.csv")
    for lv in ("1-1", "4-1"):
        rows.append((lv, "hai màn (lịch 3M)", d["global_step"] / 2, d[f"flag_rate_{lv}"], d[f"episodes_{lv}"]))
    d8, ls8 = level_steps_8()
    for lv in LEVELS8:
        rows.append((lv, "tám màn (lịch 10M)", ls8[lv], d8[f"flag_rate_{lv}"], d8[f"episodes_{lv}"]))
    d = load(RUNS / "2-3/from-scratch-3m/progress.csv")
    rows.append(("2-3", "một màn (lịch 3M)", d["global_step"], d["flag_rate"], d["episodes"]))
    d = load(RUNS / "2-3/from-scratch/progress.csv")
    rows.append(("2-3", "một màn (lịch 1,2M)", d["global_step"], d["flag_rate"], d["episodes"]))
    for lv, setting, s, r, ep in sorted(rows, key=lambda x: x[0]):
        a, b = first_cross(s, r, 0.5, ep), first_cross(s, r, 0.8, ep)
        fa = f"{a / 1e6:4.2f}M" if np.isfinite(a) else "  chưa"
        fb = f"{b / 1e6:4.2f}M" if np.isfinite(b) else "  chưa"
        print(f"  {lv} {setting:30s}: 50% ở {fa}, 80% ở {fb} bước riêng (tổng bước riêng {s[-1] / 1e6:.2f}M)")

    print("\n== B2. Khớp logistic vào các lần chấm 100 lượt (seed 7): p(s) = A / (1 + exp(-(s - s0)/w))")
    fits = [("4-1 một màn", RUNS / "4-1/from-scratch/eval", 1), ("4-1 trong hai màn", RUNS / "1-1+4-1/from-scratch/eval/4-1", 2),
            ("1-1 trong hai màn", RUNS / "1-1+4-1/from-scratch/eval/1-1", 2),
            ("2-3, lịch 1,2M", RUNS / "2-3/from-scratch/eval", 1), ("2-3, lịch 3M", RUNS / "2-3/from-scratch-3m/eval", 1)]
    fig, axes = new_fig(2)
    for i, (name, folder, share) in enumerate(fits):
        g, k = eval_series(folder)
        s = g / share
        A, s0, w = logistic_fit(s, k, 100)
        s80 = s0 + w * np.log(0.8 / (A - 0.8)) if A > 0.8 else np.nan
        print(f"  {name:18s}: A = {A:.3f}, s0 = {s0:.2f}M bước riêng, w = {w * 1e3:.0f}k; "
              f"tới 80%: {s80:.2f}M bước riêng" if np.isfinite(s80) else
              f"  {name:18s}: A = {A:.3f} (< 0,8), s0 = {s0:.2f}M, w = {w * 1e3:.0f}k")
        if "2-3" in name:  # logistic không hợp với 2-3 (lượt 1,2M bị cắt khi còn đang lên): nối điểm
            c = SERIES[i - 3]
            axes[1].plot(s / 1e6, k / 100, color=c, lw=1.5, marker="o", ms=3.5, label=name)
            continue
        c = SERIES[i]
        axes[0].scatter(s / 1e6, k / 100, s=12, color=c, alpha=0.8, linewidths=0)
        xs = np.linspace(0, s.max() / 1e6, 200)
        axes[0].plot(xs, A / (1 + np.exp(-(xs - s0) / w)), color=c, lw=2,
                     label=f"{name}: A {A:.2f}, s0 {s0:.2f}M".replace(".", ","))
    notes = ("chấm: chạm cờ / 100 lượt (seed 7); đường: logistic khớp hợp lý cực đại",
             "chạm cờ / 100 lượt (seed 7); tốc độ học về 0 ở 1,2M và ở 3M")
    for ax, t, note in ((axes[0], "1-1 và 4-1: một màn so với hai màn", notes[0]),
                        (axes[1], "2-3: cùng cấu hình, hai độ dài lịch", notes[1])):
        ax.axhline(0.8, color=MUTED, lw=1, ls="--")
        ax.set_xlabel("triệu bước riêng của màn", fontsize=8, color=MUTED)
        ax.set_ylim(-0.02, 1.05)
        style(ax, t, note)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "learning-curves.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    print("\n== B3. Lượt 8 màn: độ dốc cuối lượt (25% số bước cuối) của tỉ lệ chạm cờ lúc huấn luyện theo bước riêng")
    a = load(RUNS / "8-levels/from-scratch-10m/allocation.csv")
    tail = d8["global_step"] >= 7.5e6
    a_tail = a["global_step"] >= 7.5e6
    slopes, shares = [], []
    for lv in LEVELS8:
        x, y = ls8[lv][tail], d8[f"flag_rate_{lv}"][tail]
        slope = np.polyfit(x / 1e5, y, 1)[0] * 100  # điểm phần trăm mỗi 100 nghìn bước riêng
        share = a[f"steps_{lv}"][a_tail].sum() / a["steps_1-1"][a_tail].size / 4096
        slopes.append(slope)
        shares.append(share)
        print(f"  {lv}: dốc {slope:+5.2f} điểm/100k bước riêng; phần bước ở đoạn này {100 * share:4.1f}%; "
              f"chạm cờ lúc huấn luyện cuối {d8[f'flag_rate_{lv}'][-1]:.2f}")
    fig, axes = new_fig(2, h=3.8)
    order = np.argsort(slopes)[::-1]
    y = np.arange(len(LEVELS8))
    axes[0].barh(y, np.array(slopes)[order], color=SERIES[0], height=0.6)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color=MUTED, lw=1)
    axes[0].set_xlabel("điểm phần trăm mỗi 100 nghìn bước riêng", fontsize=8, color=MUTED)
    style(axes[0], "Lợi ích biên ở 2,5 triệu bước cuối", "độ dốc tỉ lệ chạm cờ lúc huấn luyện theo bước riêng")
    axes[1].barh(y, 100 * np.array(shares)[order], color=SERIES[1], height=0.6)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("% số bước của đoạn 7,5–10 triệu", fontsize=8, color=MUTED)
    style(axes[1], "Phần bước thật sự được chia", "cùng thứ tự màn với hình trái")
    for ax in axes:  # sau style(): style() thay bộ định dạng trục
        ax.set_yticks(y, [LEVELS8[j] for j in order])
    for ax, vals, fmt in ((axes[0], np.array(slopes)[order], "{:+.1f}"), (axes[1], 100 * np.array(shares)[order], "{:.0f}%")):
        for yy, v in zip(y, vals):
            ax.text(max(v, 0), yy, " " + fmt.format(v).replace(".", ","), va="center", fontsize=8, color=INK2,
                    ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "marginal-gain.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------
# C. Thống kê
# ------------------------------------------------------------------------------------------------------------
def statistics():
    print("\n== C1. Độ phân tán giữa các checkpoint liền nhau ở vùng đã phẳng (kiểm định đồng nhất χ², n = 100 mỗi điểm)")
    windows = [("1-1 trong hai màn, 2,1–3,0M", RUNS / "1-1+4-1/from-scratch/eval/1-1", 2.1e6),
               ("4-1 trong hai màn, 2,1–3,0M", RUNS / "1-1+4-1/from-scratch/eval/4-1", 2.1e6),
               ("2-3 lịch 3M, 2,1–3,0M", RUNS / "2-3/from-scratch-3m/eval", 2.1e6),
               ("1-1 một màn, 0,8–1,2M", RUNS / "1-1/run-1/eval", 0.8e6),
               ("4-1 một màn, 0,8–1,2M", RUNS / "4-1/from-scratch/eval", 0.8e6)]
    for name, folder, start in windows:
        g, k = eval_series(folder)
        k = k[g >= start]
        m, pbar = len(k), k.sum() / (100 * len(k))
        chi2 = ((k - 100 * pbar) ** 2 / (100 * pbar * (1 - pbar))).sum() if 0 < pbar < 1 else 0.0
        pval = stats.chi2.sf(chi2, m - 1)
        print(f"  {name:28s}: {m} checkpoint, chạm cờ {' '.join(str(int(v)) for v in k)}; p̄ = {pbar:.3f}; "
              f"χ²/df = {chi2 / (m - 1):.2f}, p = {pval:.3f}")
        # lời nguyền của người thắng: kỳ vọng max của m checkpoint đồng nhất, độ lệch so với trung bình
        sims = stats.binom.rvs(100, pbar, size=(20000, m), random_state=1).max(axis=1)
        print(f"      nếu mọi checkpoint như nhau: checkpoint cao nhất trung bình {sims.mean():.1f}/100, "
              f"cao hơn tỉ lệ thật {sims.mean() - 100 * pbar:+.1f} điểm")

    print("\n== C2. Số lượt mỗi nhánh để phát hiện khác biệt (kiểm định hai tỉ lệ, α = 0,05 hai phía, lực 0,8)")
    za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
    for p1, p2 in [(0.80, 0.90), (0.85, 0.95), (0.70, 0.80), (0.10, 0.40), (0.10, 0.20)]:
        n = (za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2
        print(f"  {p1:.2f} so với {p2:.2f}: {int(np.ceil(n))} lượt mỗi nhánh")
    print("  Khoảng tin cậy Wilson 95% cho lệnh chấm 20 lượt:")
    for k in (16, 18, 19, 20):
        lo, hi = wilson(k, 20)
        print(f"    {k}/20: {100 * lo:.0f}–{100 * hi:.0f}%")


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


# ------------------------------------------------------------------------------------------------------------
# D. Entropy
# ------------------------------------------------------------------------------------------------------------
def entropy_math():
    print("\n== D1. Cân bằng của phần thưởng entropy: π(a) ∝ exp(Â(a)/c), Â là lợi thế đã chuẩn hóa (độ lệch chuẩn 1)")
    for c in (0.01, 0.002):
        ratios = ", ".join(f"Δ = {dlt}: {np.exp(-dlt / c):.2g}" for dlt in (0.01, 0.05, 0.1))
        print(f"  c = {c}: tỉ lệ xác suất của hành động kém hơn Δ độ lệch chuẩn so với hành động tốt nhất: {ratios}")
    print("\n== D2. Entropy của chính sách (p_max, phần còn lại chia đều cho 6 hành động) — để đọc entropy trung bình")
    for pm in (0.5, 0.7, 0.8, 0.9, 0.95, 0.97):
        h = -pm * np.log(pm) - (1 - pm) * np.log((1 - pm) / 6)
        print(f"  p_max = {pm:.2f}: H = {h:.2f} nat (tối đa ln 7 = {np.log(7):.2f})")
    print("\n== D3. Mô hình 'K quyết định then chốt': lấy mẫu / tham lam = p^K")
    for name, greedy, sampled in [("5-2, lượt 8 màn", 0.87, 0.735)]:
        for p in (0.9, 0.95, 0.98):
            print(f"  {name}: {sampled}/{greedy} = {sampled / greedy:.3f} -> K = {np.log(sampled / greedy) / np.log(p):.1f} "
                  f"quyết định nếu mỗi quyết định có p = {p}")


# ------------------------------------------------------------------------------------------------------------
# E. Hệ số chiết khấu
# ------------------------------------------------------------------------------------------------------------
def discount_figure():
    fig, axes = new_fig(1, w=8.5, h=4.0)
    ax = axes[0]
    k = np.arange(0, 81)
    ax.axvspan(7, 15, color=SERIES[0], alpha=0.10, lw=0)
    ax.axvspan(23, 26, color=SERIES[1], alpha=0.18, lw=0)
    ax.text(11, 1.02, "critic thấy trước\n(7–15 bước)", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.text(24.5, 1.02, "kết cục đã định\n(23–26 bước)", ha="center", va="bottom", fontsize=8, color=INK2)
    for i, g in enumerate((0.9, 0.95, 0.97, 0.99)):
        ax.plot(k, g ** k, color=SERIES[i], lw=2, label=f"γ = {str(g).replace('.', ',')}")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    ax.axhline(0.1, color=MUTED, lw=1, ls="--")
    ax.text(0.5, 0.115, "trọng số 10%", fontsize=8, color=MUTED)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("số bước tác tử tới sự kiện (1 bước = 1/15 giây)", fontsize=8, color=MUTED)
    style(ax, "Trọng số γ^k của một phần thưởng cách k bước",
          "vùng màu: số đo của docs/inside-the-network.md ở checkpoint cuối của lượt 8 màn")
    fig.tight_layout()
    fig.savefig(FIG / "discount.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------
# F. Thang nhiễu gradient
# ------------------------------------------------------------------------------------------------------------
def noise_scale():
    files = sorted((RUNS / "1-1/run-1/eval/gradient-noise").glob("*.json")) + \
        sorted((RUNS / "8-levels/from-scratch-10m/eval/gradient-noise").glob("*.json"))
    if not files:
        print("\n== F. chưa có kết quả của experiments/gradient_noise.py")
        return
    print("\n== F. Thang nhiễu gradient tại θ của checkpoint (B_simple = tr Σ / |G|²)")
    labels, between, within = [], [], []
    for f in files:
        r = json.loads(f.read_text())
        run = "1-1" if "1-1/run-1" in r["ckpt"] else "8 màn"
        p, t = r["policy"], r["total"]
        sig = 1 / (1 + p["rollout_noise_to_signal"])  # phần tín hiệu trong |G_đoạn|² của gradient chính sách
        print(f"  {run:5s} bước {r['global_step'] / 1e6:5.2f}M (chạm cờ lúc đo {r['train_flag_rate']:.2f}, "
              f"{r['episodes']} lượt): CHÍNH SÁCH B_simple trong đoạn = {p['B_simple_within_rollout']:,.0f}, "
              f"giữa các đoạn = {p['B_simple_between_rollouts']:,.0f}, nhiễu/tín hiệu của cả đoạn "
              f"{p['rollout_noise_to_signal']:.1f} (tín hiệu {100 * sig:.0f}%), cos giữa các đoạn "
              f"{p['cos_between_rollouts']:+.3f} | TỔNG (có critic) B_simple trong đoạn = "
              f"{t['B_simple_within_rollout']:,.0f}, giữa các đoạn = {t['B_simple_between_rollouts']:,.0f}, "
              f"cos {t['cos_between_rollouts']:+.3f}")
        labels.append(f"{run}\n{r['global_step'] / 1e6:.1f}M".replace(".", ","))
        between.append(p["B_simple_between_rollouts"])
        within.append(p["B_simple_within_rollout"])
    for f in sorted((RUNS / "1-1/run-1/eval/gradient-noise/16x256").glob("*.json")):
        r = json.loads(f.read_text())
        base = json.loads((RUNS / "1-1/run-1/eval/gradient-noise" / f.name).read_text())
        for k in ("policy", "total"):
            print(f"  16x256 so với 8x512, 1-1 bước {r['global_step'] / 1e6:.2f}M, {k}: nhiễu/tín hiệu của cả đoạn "
                  f"{r[k]['rollout_noise_to_signal']:.2f} so với {base[k]['rollout_noise_to_signal']:.2f}; "
                  f"B_simple giữa các đoạn {r[k]['B_simple_between_rollouts']:,.0f} so với "
                  f"{base[k]['B_simple_between_rollouts']:,.0f}; cos {r[k]['cos_between_rollouts']:+.3f} so với "
                  f"{base[k]['cos_between_rollouts']:+.3f}")

    fig, axes = new_fig(1, w=8.5, h=4.0)
    ax = axes[0]
    x = np.arange(len(labels))
    between = np.array(between, dtype=float)
    fin = np.isfinite(between) & (between > 0)
    ax.scatter(x[fin], between[fin], s=60, color=SERIES[0], zorder=3, label="giữa các đoạn thu")
    if (~fin).any():
        # Không đo được tín hiệu với 8 đoạn: |G|² ước lượng ≤ 0, tức B_simple > 4096 x 8 (cận dưới thô).
        ax.scatter(x[~fin], np.full((~fin).sum(), 4096 * 8), s=70, marker="^", color=SERIES[0], zorder=3,
                   label="giữa các đoạn: không đo được tín hiệu, chỉ có cận dưới 4.096 × 8")
    ok = np.array(within) > 0
    ax.scatter(x[ok], np.array(within)[ok], s=60, facecolors="none", edgecolors=SERIES[0], linewidths=1.5,
               zorder=3, label="trong một đoạn thu (không có điểm: tín hiệu dưới mức đo được)")
    for yv, txt in ((4096, "một đoạn thu: 4.096 mẫu"), (256, "một mini-batch: 256 mẫu")):
        ax.axhline(yv, color=MUTED, lw=1, ls="--")
        ax.text(len(labels) - 0.6, yv * 1.15, txt, fontsize=8, color=MUTED, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(100, 1e6)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    style(ax, "Thang nhiễu của gradient chính sách, B_simple = tr Σ / |G|²",
          "số mẫu cần để tín hiệu bằng nhiễu; đo tại θ của checkpoint, 8 đoạn thu mỗi checkpoint")
    ax.set_xticks(x, labels)  # đặt lại sau style(), vì style() thay bộ định dạng trục ngang
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left", bbox_to_anchor=(0, 0.13))
    fig.tight_layout()
    fig.savefig(FIG / "noise-scale.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    trust_region()
    learning_curves()
    statistics()
    entropy_math()
    discount_figure()
    noise_scale()
