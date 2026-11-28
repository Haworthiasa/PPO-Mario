"""Tổng hợp đợt tối ưu cấu hình ở 1-1 (docs/optimize-1-1.md): đọc kết quả chấm của mỗi nhánh trong runs/1-1/tuning,
tính hai thước đo đã chốt trước và hiệu ứng chính của từng yếu tố trên những nhánh đã có đủ số liệu.

  Y = tỉ lệ chạm cờ của checkpoint cuối trên 200 lượt chấm (100 lượt seed 7 + 100 lượt seed 11)
  Z = trung bình tỉ lệ chạm cờ seed 7 trên 6 checkpoint (102.400 ... 614.400 bước)

Hiệu ứng chính của yếu tố X = trung bình (Y hoặc Z) của các nhánh có X = 1 trừ trung bình của các nhánh có X = 0.
Khi mới có nửa đầu (4 nhánh đều C = 0) thì không tính được C, và D trùng với −A·B (xem Mục 7 của tài liệu).

Chạy: python experiments/tuning_summary.py [--root runs/1-1/tuning] [--json out.json] [--plot hình.png]
Với --plot, vẽ số chấm seed 7 theo bước của mọi nhánh, kèm lượt đã nộp (runs/1-1/run-1) để so.
"""
import argparse
import csv
import json
import math
from pathlib import Path

FINAL = 614_400
STEPS = [102_400 * k for k in range(1, 7)]
FACTORS = "abcd"


def levels(arm):
    """'a1-b0-c0-d1' -> {'a': 1, 'b': 0, 'c': 0, 'd': 1}"""
    return {part[0]: int(part[1]) for part in arm.split("-")}


def flag(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["flag_rate"], d["episodes"]


def train_stats(arm_dir):
    """Tốc độ trung bình và tỉ lệ chạm cờ lúc huấn luyện ở đoạn cuối, từ progress.csv."""
    rows = list(csv.DictReader(open(arm_dir / "progress.csv", encoding="utf-8")))
    if not rows:
        return None
    last = rows[-1]
    sps = sum(float(r["sps"]) for r in rows[-10:]) / min(10, len(rows))
    return {"train_flag_last": float(last["flag_rate"]), "sps_last10": round(sps, 1), "global_step": int(last["global_step"])}


def summarize_arm(arm_dir):
    ev = arm_dir / "eval"
    s7 = {}
    for step in STEPS:
        p = ev / f"100-seed7_{step:09d}.json"
        if p.exists():
            s7[step] = flag(p)[0]
    p11 = ev / f"100-seed11_{FINAL:09d}.json"
    out = {"arm": arm_dir.name, "seed7": s7, "train": train_stats(arm_dir) if (arm_dir / "progress.csv").exists() else None}
    out["Z"] = sum(s7.values()) / len(s7) if len(s7) == len(STEPS) else None
    if FINAL in s7 and p11.exists():
        out["seed11_final"] = flag(p11)[0]
        out["Y"] = (s7[FINAL] + out["seed11_final"]) / 2
    else:
        out["Y"] = None
    return out


def effects(arms, key):
    """Hiệu ứng chính của mỗi yếu tố có cả hai mức trong số các nhánh đã có `key`."""
    done = [a for a in arms if a[key] is not None]
    res = {}
    for f in FACTORS:
        hi = [a[key] for a in done if levels(a["arm"])[f] == 1]
        lo = [a[key] for a in done if levels(a["arm"])[f] == 0]
        if hi and lo:
            res[f.upper()] = {"effect": sum(hi) / len(hi) - sum(lo) / len(lo), "n_hi": len(hi), "n_lo": len(lo)}
    return res


LEVEL_NAMES = {"a": "tốc độ học 4e-4", "b": "γ 0,97", "c": "λ 0,95", "d": "16 × 256"}
OLD_RUN = Path("runs/1-1/run-1")
COLORS = ["#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def arm_label(arm):
    on = [LEVEL_NAMES[f] for f, v in levels(arm).items() if v == 1]
    return ", ".join(on) if on else "đối chứng (cấu hình nộp, lịch 614k)"


def plot(arms, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = []
    old = sorted((int(p.stem.split("_")[1]), flag(p)[0]) for p in (OLD_RUN / "eval").glob("100-seed7_*.json"))
    if old:
        series.append(("Bản nộp run-1 (lịch 2M, dừng ở 1,2M)", old, "solid"))
    for a in arms:
        pts = sorted(a["seed7"].items())
        if pts:
            series.append((f"{a['arm']}: {arm_label(a['arm'])}", pts, "solid"))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    for i, (label, pts, style) in enumerate(series):
        ax.plot([s / 1e3 for s, _ in pts], [100 * v for _, v in pts], color=COLORS[i], linestyle=style,
                linewidth=2 if i else 1.6, marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=1,
                label=label)
    ax.axhline(80, color="#898781", linestyle=(0, (1, 3)), linewidth=1, label="Ngưỡng vượt màn 80%")
    ax.axvline(FINAL / 1e3, color="#c3c2b7", linewidth=1)
    ax.text(FINAL / 1e3 + 10, 95, "ngân sách của đợt: 614.400 bước", color="#898781", fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(max(s for s, _ in pts) for _, pts, _ in series) / 1e3 + 20)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("Bước (nghìn)", color="#52514e")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", length=0)
    ax.set_title("Màn 1-1: tỉ lệ chạm cờ khi chấm (evaluate.py, 100 lượt seed 7 mỗi checkpoint)", loc="left",
                 fontsize=10.5, color="#2b2a27")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("đã ghi", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/1-1/tuning")
    ap.add_argument("--json", default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    arms = [summarize_arm(d) for d in sorted(root.iterdir()) if d.is_dir() and d.name.startswith("a")]

    print(f"{'nhánh':<14}" + "".join(f"{s // 1000:>7}k" for s in STEPS) + f"{'seed11':>8}{'Y':>7}{'Z':>7}{'cờ-HL':>7}{'sps':>7}")
    for a in arms:
        row = "".join(f"{a['seed7'][s]:>8.2f}" if s in a["seed7"] else f"{'-':>8}" for s in STEPS)
        s11 = f"{a['seed11_final']:>8.2f}" if a.get("seed11_final") is not None else f"{'-':>8}"
        y = f"{a['Y']:>7.3f}" if a["Y"] is not None else f"{'-':>7}"
        z = f"{a['Z']:>7.3f}" if a["Z"] is not None else f"{'-':>7}"
        t = a["train"] or {}
        print(f"{a['arm']:<14}{row}{s11}{y}{z}{t.get('train_flag_last', float('nan')):>7.2f}{t.get('sps_last10', float('nan')):>7.1f}")

    out = {"arms": arms}
    for key in ("Y", "Z"):
        eff = effects(arms, key)
        out[f"effects_{key}"] = eff
        if eff:
            # sai số nhị thức của hiệu ứng: mỗi nhánh có 200 (Y) hoặc 600 (Z) lượt chấm; chưa tính phần lệch giữa seed
            n_ep = 200 if key == "Y" else 600
            vals = [a[key] for a in arms if a[key] is not None]
            p = sum(vals) / len(vals)
            print(f"\nHiệu ứng chính trên {key} (sai số chỉ tính phần nhị thức):")
            for f, e in eff.items():
                se = math.sqrt(p * (1 - p) / n_ep * (1 / e["n_hi"] + 1 / e["n_lo"]))
                e["se_binomial"] = se
                print(f"  {f}: {e['effect']:+.3f}  ± {se:.3f}   ({e['n_hi']} nhánh mức 1, {e['n_lo']} nhánh mức 0)")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    if args.plot:
        plot(arms, args.plot)


if __name__ == "__main__":
    main()
