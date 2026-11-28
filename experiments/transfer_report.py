"""Báo cáo đợt 3: cấu hình được chọn ở 1-1 mang sang 4-1 và 2-3 (docs/optimize-1-1.md, Mục 9). Chỉ đọc tệp.

    python experiments/transfer_report.py [--plot hình.png] [--json tổng.json] [--predict]

Mô hình quãng đường: M(bước) = tổng √approx_kl qua các lần cập nhật đã qua (mỗi lần cập nhật 4.096 mẫu). Giả thuyết:
ở một màn, tỉ lệ chạm cờ chủ yếu phụ thuộc M, không phụ thuộc cấu hình đã sinh ra M. Đường "chạm cờ theo M" của mỗi
màn được khớp từ các lượt cũ của màn đó bằng hồi quy đơn điệu (isotonic). --predict in dự đoán ghi trước của Mục 9.
"""
import argparse
import csv
import glob
import json
import math
from pathlib import Path

from tuning_summary import flag

LEVELS = {
    "4-1": {"new": Path("runs/4-1/tuning/best-seed1"), "final": 614_400,
            "old": {"cũ, 1,2M bước (bản nộp)": Path("runs/4-1/from-scratch")}},
    "2-3": {"new": Path("runs/2-3/tuning/best-seed1"), "final": 1_228_800,
            "old": {"cũ, 1,2M bước": Path("runs/2-3/from-scratch"),
                    "cũ, 3M bước (bản nộp)": Path("runs/2-3/from-scratch-3m")}},
}
# Dự đoán ghi trước cho cấu hình thực sự được chọn (c1-e0-f1-g0, Mục 9.5): Y ở checkpoint cuối, khoảng 80%.
# (Dự đoán ở Mục 9.2 tính cho lượt đổi cả bốn tham số, đã bỏ vì luật chọn ra cấu hình khác.)
PREDICTED = {"4-1": (0.72, 0.55, 0.90), "2-3": (0.66, 0.52, 0.80)}
# Tốc độ tích M giả định khi ghi dự đoán ở Mục 9.2 (cho lượt đổi cả bốn tham số): lần cập nhật đầu tiên tới 102.400
# bước, rồi mỗi 102.400 bước. Chỉ dùng cho --predict; Mục 9.5 tính lại M cho cấu hình đã chọn.
M_FIRST, M_RATE = 2.9, 3.7
# Vùng chết chính của mỗi màn (Mục 9.3, experiments/level_map.py) và x của cờ
ZONES = {"4-1": [(384, 576), (1248, 1344), (1760, 1952), (1984, 2272), (2368, 2496), (2592, 2720), (2784, 2944),
                 (3008, 3136)],
         "2-3": [(1056, 1248), (1312, 1408), (1536, 1600), (1664, 2080), (2080, 2304), (2304, 2592), (2624, 3040),
                 (3136, 3360)]}
FLAG_X = 3593


def zone_deaths(level, files):
    """[(x0, x1, chết, tới được)] của các vùng chính, gộp các tệp chấm cho trước."""
    xs = []
    for f in files:
        xs += json.load(open(f, encoding="utf-8"))["x_pos"]
    return [(a, b, sum(1 for x in xs if a <= x < b and x < FLAG_X), sum(1 for x in xs if x >= a))
            for a, b in ZONES[level]]


def movement(run):
    """[(bước, M)] từ progress.csv."""
    out, m = [], 0.0
    with open(run / "progress.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m += math.sqrt(max(float(r["approx_kl"]), 0.0))
            out.append((int(r["global_step"]), m))
    return out


def m_at(ms, step):
    return min(ms, key=lambda p: abs(p[0] - step))[1]


def seed7(run):
    pts = []
    for f in sorted(glob.glob(str(run / "eval" / "100-seed7_*.json"))):
        pts.append((int(Path(f).stem.split("_")[-1]), flag(f)[0]))
    return pts


def isotonic(xs, ys):
    """Hồi quy đơn điệu tăng (pool-adjacent-violators). Trả [(x, y_khớp)] theo x tăng."""
    pts = sorted(zip(xs, ys))
    blocks = [[y, 1, [x]] for x, y in pts]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v1, n1, x1 = blocks[i]
            v2, n2, x2 = blocks.pop(i + 1)
            blocks[i] = [(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2, x1 + x2]
            i = max(i - 1, 0)
        else:
            i += 1
    return [(x, b[0]) for b in blocks for x in b[2]]


def curve_of_level(level):
    """Đường chạm cờ theo M khớp từ mọi lượt cũ của màn."""
    xs, ys = [], []
    for run in LEVELS[level]["old"].values():
        ms = movement(run)
        for step, v in seed7(run):
            xs.append(m_at(ms, step))
            ys.append(v)
    return isotonic(xs, ys)


def interp(curve, m):
    if m <= curve[0][0]:
        return curve[0][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= m <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (m - x0) / (x1 - x0)
    return curve[-1][1]


def mechanism(run, final):
    rows = list(csv.DictReader(open(run / "progress.csv", encoding="utf-8")))
    bands = []
    for k in range(0, final, 102_400):
        b = [r for r in rows if k < int(r["global_step"]) <= k + 102_400]
        if b:
            bands.append({"to": k + 102_400, "kl": sum(float(r["approx_kl"]) for r in b) / len(b),
                          "lr": sum(float(r["lr"]) for r in b) / len(b),
                          "entropy": sum(float(r["entropy"]) for r in b) / len(b)})
    return {"bands": bands, "entropy_end": sum(float(r["entropy"]) for r in rows[-10:]) / min(10, len(rows)),
            "train_flag_end": float(rows[-1]["flag_rate"]), "steps": int(rows[-1]["global_step"])}


def summarize(level):
    spec = LEVELS[level]
    run, final = spec["new"], spec["final"]
    curve = curve_of_level(level)
    out = {"level": level, "run": str(run), "final": final, "predicted": PREDICTED[level]}
    if not (run / "progress.csv").exists():
        return out
    ms = movement(run)
    s7 = seed7(run)
    out["seed7"] = s7
    out["M"] = [(s, m_at(ms, s)) for s, _ in s7]
    # dự đoán của mô hình quãng đường từ M THỰC TẾ của lượt mới (tách phần "đường theo M" khỏi phần "tốc độ tích M")
    out["model_from_actual_M"] = [(s, interp(curve, m)) for s, m in out["M"]]
    p11 = run / "eval" / f"100-seed11_{final:09d}.json"
    last7 = dict(s7).get(final)
    out["seed11_final"] = flag(p11)[0] if p11.exists() else None
    out["Y"] = (last7 + out["seed11_final"]) / 2 if last7 is not None and out["seed11_final"] is not None else None
    finals = [p for p in (run / "eval" / f"100-seed7_{final:09d}.json", p11) if p.exists()]
    out["zones_final"] = zone_deaths(level, finals) if finals else None
    out["mechanism"] = mechanism(run, final)
    return out


def predict():
    """Dự đoán ghi trước: M giả định -> đường theo M của màn."""
    for level, spec in LEVELS.items():
        curve = curve_of_level(level)
        n = spec["final"] // 102_400
        line = []
        for k in range(1, n + 1):
            m = M_FIRST + M_RATE * (k - 1)
            line.append(f"{k * 102.4:.0f}k: M {m:.1f} -> {100 * interp(curve, m):.0f}%")
        m_end = M_FIRST + M_RATE * (n - 1)
        lo = interp(curve, M_FIRST + 0.85 * M_RATE * (n - 1))
        hi = interp(curve, M_FIRST + 1.15 * M_RATE * (n - 1))
        print(f"{level}: " + "; ".join(line))
        print(f"   cuối: M {m_end:.1f} -> {100 * interp(curve, m_end):.0f}% (tốc độ tích M ±15%: {100 * lo:.0f}–{100 * hi:.0f}%)")
        print("   đường theo M (khớp đơn điệu):", ", ".join(f"{x:.1f}:{100 * y:.0f}" for x, y in curve))


def plot(summaries, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.4), dpi=150)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.07, hspace=0.42, wspace=0.18)
    fig.suptitle("Cấu hình chọn ở 1-1 mang sang 4-1 và 2-3: tỉ lệ chạm cờ khi chấm (100 lượt seed 7 mỗi checkpoint)",
                 x=0.02, ha="left", fontsize=11, color="#2b2a27")
    grays = [(0, (4, 2)), (0, (1, 2))]
    for col, s in enumerate(summaries):
        level = s["level"]
        top, bot = axes[0][col], axes[1][col]
        for (name, run), style in zip(LEVELS[level]["old"].items(), grays):
            pts = seed7(run)
            ms = movement(run)
            top.plot([p / 1e3 for p, _ in pts], [100 * v for _, v in pts], color="#a3a198", linestyle=style,
                     linewidth=1.4, label=name)
            bot.plot([m_at(ms, p) for p, _ in pts], [100 * v for _, v in pts], color="#a3a198", linestyle=style,
                     linewidth=1.2, marker="o", markersize=2.5, label=name)
        curve = curve_of_level(level)
        bot.plot([x for x, _ in curve], [100 * y for _, y in curve], color="#52514e", linewidth=1.0,
                 drawstyle="steps-post", alpha=0.6, label="đường khớp đơn điệu")
        if s.get("seed7"):
            top.plot([p / 1e3 for p, _ in s["seed7"]], [100 * v for _, v in s["seed7"]], color="#eb6834",
                     linewidth=2.4, marker="o", markersize=4, markeredgecolor="white", label="cấu hình mới, seed 1")
            bot.plot([m for _, m in s["M"]], [100 * v for _, v in s["seed7"]], color="#eb6834", linewidth=2.2,
                     marker="o", markersize=4, markeredgecolor="white", label="cấu hình mới, seed 1")
        p, lo, hi = s["predicted"]
        x_end = s["final"] / 1e3
        top.plot([x_end, x_end], [100 * lo, 100 * hi], color="#c3c2b7", linewidth=3, solid_capstyle="round")
        top.plot([x_end], [100 * p], marker="o", markersize=7, markerfacecolor="white", markeredgecolor="#898781",
                 markeredgewidth=1.6, linestyle="none", label="dự đoán ghi trước (Y cuối)")
        if s.get("Y") is not None:
            top.plot([x_end], [100 * s["Y"]], marker="o", markersize=7, color="#eb6834", linestyle="none")
            top.annotate(f"Y {100 * s['Y']:.1f}%", (x_end, 100 * s["Y"]), textcoords="offset points", xytext=(10, -4),
                         ha="left", fontsize=8.5, color="#2b2a27")
        for ax in (top, bot):
            ax.axhline(80, color="#898781", linestyle=(0, (1, 3)), linewidth=1)
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 20))
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
            ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color("#c3c2b7")
            ax.tick_params(colors="#898781", length=0)
        top.set_xlim(0, max(3000 if level == "2-3" else 1250, x_end + 30))
        top.set_xlabel("Bước (nghìn)", color="#52514e")
        top.set_title(f"Màn {level}: theo số bước", loc="left", fontsize=10, color="#2b2a27")
        bot.set_xlabel("Quãng đường M = tổng √KL của các lần cập nhật", color="#52514e")
        bot.set_title(f"Màn {level}: theo quãng đường M (mô hình)", loc="left", fontsize=10, color="#2b2a27")
        top.legend(loc="upper left", frameon=False, fontsize=7.5)
        bot.legend(loc="lower right", frameon=False, fontsize=7.5)
    fig.savefig(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--predict", action="store_true")
    a = ap.parse_args()
    if a.predict:
        predict()
        return
    summaries = [summarize(level) for level in LEVELS]
    for s in summaries:
        p, lo, hi = s["predicted"]
        print(f"== {s['level']} ({s['run']}), dự đoán Y {p:.2f} ({lo:.2f}–{hi:.2f})")
        if not s.get("seed7"):
            print("   chưa có số")
            continue
        print("   seed 7:", " ".join(f"{st / 1e3:.0f}k:{100 * v:.0f}" for st, v in s["seed7"]))
        print("   M:     ", " ".join(f"{st / 1e3:.0f}k:{m:.1f}" for st, m in s["M"]))
        print("   mô hình theo M thực tế:", " ".join(f"{st / 1e3:.0f}k:{100 * v:.0f}" for st, v in s["model_from_actual_M"]))
        if s.get("Y") is not None:
            print(f"   seed 11 cuối {100 * s['seed11_final']:.0f}, Y = {s['Y']:.3f}")
        if s.get("zones_final"):
            print("   chết / tới được theo vùng ở checkpoint cuối:",
                  "  ".join(f"{a}–{b}: {d}/{r}" for a, b, d, r in s["zones_final"]))
        mech = s["mechanism"]
        print("   KL theo đoạn 102k:", " ".join(f"{b['kl']:.4f}" for b in mech["bands"]))
        print("   lr theo đoạn:     ", " ".join(f"{b['lr']:.1e}" for b in mech["bands"]))
        print(f"   entropy cuối {mech['entropy_end']:.3f}, chạm cờ lúc huấn luyện cuối {mech['train_flag_end']:.2f}, "
              f"tới bước {mech['steps']}")
    if a.plot:
        plot(summaries, a.plot)
        print("đã ghi", a.plot)
    if a.json:
        Path(a.json).write_text(json.dumps(summaries, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
