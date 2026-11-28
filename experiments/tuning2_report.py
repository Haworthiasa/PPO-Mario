"""Báo cáo đợt 2 tối ưu cấu hình ở 1-1 (docs/optimize-1-1.md, Mục 8): số chấm của các lượt đã xong, so với dự đoán
ghi trước ở Mục 8.1, kèm các số kiểm tra cơ chế đọc từ progress.csv. Chỉ đọc tệp, không huấn luyện.

    python experiments/tuning2_report.py [--plot hình.png] [--json tổng.json]

Hiệu ứng chính chỉ được tính khi đủ 8 lượt của thiết kế; trước đó chỉ in từng lượt.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from tuning_summary import FINAL, STEPS, summarize_arm

ROOT = Path("runs/1-1/tuning-2")
STAGE1 = Path("runs/1-1/tuning")
ARMS = ["c0-e0-f0-g0", "control", "c1-e1-f1-g1", "c1-e0-f0-g1", "c0-e1-f0-g1", "c0-e0-f1-g1",
        "c1-e1-f0-g0", "c1-e0-f1-g0", "c0-e1-f1-g0"]
DESIGN = [a for a in ARMS if a != "control"]
FACTORS = {"c": "λ 0,95", "e": "trần cuối 1e-4", "f": "entropy giảm", "g": "trần đầu 6e-4"}
SHORT = {"c": "λ", "e": "trần cuối", "f": "entropy", "g": "trần đầu"}
# Dự đoán ghi trước (docs/optimize-1-1.md, Mục 8.1); khoảng 80% là ± 0,08 (đối chứng và W ghi riêng)
PREDICTED = {"c0-e0-f0-g0": 0.75, "control": 0.62, "c1-e1-f1-g1": 0.865, "c1-e0-f0-g1": 0.81, "c0-e1-f0-g1": 0.80,
             "c0-e0-f1-g1": 0.805, "c1-e1-f0-g0": 0.83, "c1-e0-f1-g0": 0.805, "c0-e1-f1-g0": 0.835}
PRED_EFFECTS = {"C": 0.03, "E": 0.04, "F": 0.025, "G": 0.015}
COLORS = {"control": "#52514e", "c0-e0-f0-g0": "#2a78d6", "c1-e1-f1-g1": "#eb6834", "c1-e0-f0-g1": "#1baf7a",
          "c0-e1-f0-g1": "#eda100", "c0-e0-f1-g1": "#e87ba4", "c1-e1-f0-g0": "#008300", "c1-e0-f1-g0": "#6250d6",
          "c0-e1-f1-g0": "#e34948"}


def levels(arm):
    return {p[0]: int(p[1]) for p in arm.split("-")}


def label(arm, short=False):
    if arm == "control":
        return "đối chứng (cấu hình nộp)" if short else "đối chứng (cấu hình nộp), seed 2"
    on = [(SHORT if short else FACTORS)[f] for f, v in levels(arm).items() if v]
    if short:
        return "W + " + " + ".join(on) if on else "W"
    return f"{arm}: " + (" + ".join(on) if on else "W, seed 2")


def mechanism(arm_dir, arm):
    """Số kiểm tra cơ chế từ progress.csv (Mục 8.1): phần cập nhật nằm ở trần, KL và tốc độ học đoạn cuối, entropy cuối."""
    p = arm_dir / "progress.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        return None
    batch = 4096
    out = {"updates": len(rows)}
    if arm != "control":
        lv = levels(arm)
        lr_max = 6e-4 if lv["g"] else 4e-4
        cap_final = 1e-4 if lv["e"] else 0.0
        caps = [cap_final + (lr_max - cap_final) * max(0.0, 1 - (int(r["global_step"]) - batch) / FINAL) for r in rows]
        at_cap = [float(r["lr"]) >= 0.995 * c for r, c in zip(rows, caps)]
        out["at_cap"] = sum(at_cap) / len(rows)
        mid = [a for a, r in zip(at_cap, rows) if 102_400 < int(r["global_step"]) <= 512_000]
        out["at_cap_102k_512k"] = sum(mid) / len(mid) if mid else None
        last = [(a, r) for a, r in zip(at_cap, rows) if int(r["global_step"]) > 512_000]
        out["at_cap_last"] = sum(a for a, _ in last) / len(last) if last else None
    first = [r for r in rows if int(r["global_step"]) <= 102_400]
    last = [r for r in rows if int(r["global_step"]) > 512_000]
    if last:
        out["kl_last"] = sum(float(r["approx_kl"]) for r in last) / len(last)
        out["lr_last"] = sum(float(r["lr"]) for r in last) / len(last)
        s_first = (sum(float(r["approx_kl"]) for r in first) / len(first)) / \
            (sum(float(r["lr"]) ** 2 for r in first) / len(first))
        s_last = out["kl_last"] / (sum(float(r["lr"]) ** 2 for r in last) / len(last))
        out["S_growth"] = s_last / s_first
    out["entropy_end"] = sum(float(r["entropy"]) for r in rows[-10:]) / min(10, len(rows))
    out["train_flag_end"] = float(rows[-1]["flag_rate"])
    out["sps"] = sum(float(r["sps"]) for r in rows[-10:]) / min(10, len(rows))
    return out


def plot(arms, out, newest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(8, 10.5), dpi=150)
    ax = fig.add_axes([0.08, 0.54, 0.56, 0.40])  # chừa bên phải cho chú giải (tới 11 dòng khi đủ 9 lượt)
    bx = fig.add_axes([0.35, 0.05, 0.62, 0.36])
    fig.text(0.02, 0.975, "Màn 1-1, đợt 2 (seed 2): tỉ lệ chạm cờ khi chấm, 100 lượt seed 7 mỗi checkpoint",
             fontsize=11, color="#2b2a27", va="top")
    fig.text(0.02, 0.465, "Y ở bước 614.400 (200 lượt chấm): chấm đặc = thực tế, vòng rỗng = dự đoán ghi trước,\n"
             "vạch xám = khoảng 80% của dự đoán", fontsize=10, color="#2b2a27", va="top")
    # Khung trên: số chấm theo checkpoint; hai lượt của đợt 1 làm mốc (xám)
    for name, style, text in [("a1-b0-c0-d1", (0, (4, 2)), "W, seed 1 (đợt 1)"),
                              ("a0-b0-c0-d0", (0, (1, 2)), "đối chứng, seed 1 (đợt 1)")]:
        ref = summarize_arm(STAGE1 / name) if (STAGE1 / name).exists() else None
        if ref and ref["seed7"]:
            pts = sorted(ref["seed7"].items())
            ax.plot([s / 1e3 for s, _ in pts], [100 * v for _, v in pts], color="#a3a198", linestyle=style,
                    linewidth=1.4, label=text)
    for a in arms:
        pts = sorted(a["seed7"].items())
        if not pts:
            continue
        hot = a["arm"] == newest
        ax.plot([s / 1e3 for s, _ in pts], [100 * v for _, v in pts], color=COLORS[a["arm"]],
                linewidth=2.6 if hot else 1.4, alpha=1 if hot else 0.75, marker="o", markersize=5 if hot else 3.5,
                markeredgecolor="white", markeredgewidth=1,
                label=label(a["arm"], short=True) + ("\n(mới xong)" if hot else ""),
                zorder=3 if hot else 2)
    ax.axhline(80, color="#898781", linestyle=(0, (1, 3)), linewidth=1)
    ax.text(5, 81.5, "ngưỡng 80%", color="#898781", fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, FINAL / 1e3 + 15)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("Bước (nghìn)", color="#52514e")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8, handlelength=1.8,
              borderaxespad=0)

    # Khung dưới: Y thực tế (chấm đặc) so với dự đoán (vòng rỗng, vạch = khoảng 80%)
    done = {a["arm"]: a for a in arms}
    for i, arm in enumerate(ARMS):
        y = len(ARMS) - 1 - i
        pred = PREDICTED[arm]
        bx.plot([100 * (pred - 0.08), 100 * (pred + 0.08)], [y, y], color="#c3c2b7", linewidth=3, solid_capstyle="round")
        bx.plot(100 * pred, y, "o", markersize=8, markerfacecolor="white", markeredgecolor="#898781", markeredgewidth=1.5)
        a = done.get(arm)
        if a and a["Y"] is not None:
            bx.plot(100 * a["Y"], y, "o", markersize=9, color=COLORS[arm], markeredgecolor="white", markeredgewidth=1.5,
                    zorder=3)
            bx.text(100 * a["Y"], y + 0.32, f"{100 * a['Y']:.1f}%", ha="center", fontsize=8.5, color="#2b2a27")
    bx.axvline(80, color="#898781", linestyle=(0, (1, 3)), linewidth=1)
    bx.set_yticks(range(len(ARMS)))
    bx.set_yticklabels([label(a, short=True) for a in reversed(ARMS)], fontsize=8.5)
    bx.set_xlim(40, 100)
    bx.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    bx.set_ylim(-0.6, len(ARMS) - 0.4)
    bx.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    bx.spines["left"].set_visible(False)
    bx.spines["bottom"].set_color("#c3c2b7")
    bx.tick_params(colors="#52514e", length=0)
    fig.savefig(out, facecolor="white")


def effects_plot(eff, out):
    """Hiệu ứng chính và ba cặp gộp trên Y: chấm đặc = ước lượng (vạch ± 0,05, sai số gồm cả phần do seed, Mục 8.1),
    vòng rỗng = dự đoán ghi trước."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    names = [("C", "C: λ 0,95"), ("E", "E: trần cuối 1e-4"), ("F", "F: entropy 0,01 → 0,001"), ("G", "G: trần đầu 6e-4"),
             ("CE/FG", "cặp CE / FG"), ("CF/EG", "cặp CF / EG"), ("CG/EF", "cặp CG / EF")]
    pred = dict(PRED_EFFECTS, **{"CE/FG": None, "CF/EG": -0.015, "CG/EF": 0.01})
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.86, bottom=0.14)
    fig.text(0.02, 0.96, "Màn 1-1, đợt 2: hiệu ứng trên Y (8 lượt, seed 2). Chấm đặc = ước lượng, vạch = ± 0,05;\n"
             "vòng rỗng = dự đoán ghi trước", va="top", fontsize=10, color="#2b2a27")
    for i, (key, text) in enumerate(names):
        y = len(names) - 1 - i
        v = eff[key]
        ax.plot([100 * (v - 0.05), 100 * (v + 0.05)], [y, y], color="#c3c2b7", linewidth=3, solid_capstyle="round")
        ax.plot([100 * v], [y], "o", color="#2a78d6" if v >= 0 else "#e34948", markersize=7)
        ax.annotate(f"{100 * v:+.1f}", (100 * v, y), textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=8.5, color="#2b2a27")
        if pred.get(key) is not None:
            ax.plot([100 * pred[key]], [y], "o", markersize=7, markerfacecolor="white", markeredgecolor="#898781",
                    markeredgewidth=1.5)
    ax.axvline(0, color="#898781", linewidth=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([t for _, t in reversed(names)])
    ax.set_xlabel("Điểm phần trăm trên Y", color="#52514e")
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(colors="#52514e", length=0)
    fig.savefig(out)


def overrides(arm):
    """Các khóa ghi đè của một lượt thiết kế, giống experiments/run_tuning_stage2.sh nhưng không kèm seed."""
    lv = levels(arm)
    over = ["algo.num_envs=16", "algo.num_steps=256", "algo.lr_control.kl_target=0.02", "algo.lr_control.factor=1.5",
            "algo.lr_control.lr_min=1e-5", f"algo.lr_control.lr_max={'6e-4' if lv['g'] else '4e-4'}"]
    if lv["c"]:
        over.append("algo.gae_lambda=0.95")
    if lv["e"]:
        over.append("algo.lr_control.cap_final=1e-4")
    if lv["f"]:
        over.append("algo.ent_coef_final=0.001")
    return " ".join(over)


def best_overrides():
    """Luật chọn ghi trước ở Mục 9.2: lượt có Y cao nhất trong 8 lượt thiết kế. In tên lượt ra stderr, khóa ra stdout."""
    import sys
    scored = [(summarize_arm(ROOT / a)["Y"], a) for a in DESIGN if (ROOT / a).exists()]
    scored = [(y, a) for y, a in scored if y is not None]
    if len(scored) < len(DESIGN):
        sys.exit(f"mới chấm xong {len(scored)}/{len(DESIGN)} lượt thiết kế")
    y, arm = max(scored)
    print(f"chọn {arm} (Y {y:.3f})", file=sys.stderr)
    print(overrides(arm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--effects-plot", default=None, help="vẽ hiệu ứng chính khi đủ 8 lượt thiết kế")
    ap.add_argument("--best-overrides", action="store_true",
                    help="in khóa ghi đè của lượt thiết kế có Y cao nhất (luật chọn ở Mục 9.2) rồi thoát")
    args = ap.parse_args()
    if args.best_overrides:
        best_overrides()
        return
    arms = []
    for arm in ARMS:
        d = ROOT / arm
        if not d.exists():
            continue
        s = summarize_arm(d)
        s["mechanism"] = mechanism(d, arm)
        arms.append(s)
    scored = [a for a in arms if a["Y"] is not None]
    newest = None
    if scored:
        newest = max(scored, key=lambda a: (ROOT / a["arm"] / "eval" / f"100-seed11_{FINAL:09d}.json").stat().st_mtime)["arm"]

    print(f"{'lượt':<13}" + "".join(f"{s // 1000:>6}k" for s in STEPS) + f"{'s11':>6}{'Y':>7}{'dự đoán':>9}{'Z':>7}"
          f"{'ở trần':>8}{'KL cuối':>9}{'lr cuối':>9}{'entropy':>9}{'S×':>6}{'sps':>6}")
    for a in arms:
        m = a["mechanism"] or {}
        row = "".join(f"{100 * a['seed7'][s]:>7.0f}" if s in a["seed7"] else f"{'-':>7}" for s in STEPS)
        s11 = f"{100 * a['seed11_final']:>6.0f}" if a.get("seed11_final") is not None else f"{'-':>6}"
        y = f"{a['Y']:>7.3f}" if a["Y"] is not None else f"{'-':>7}"
        z = f"{a['Z']:>7.3f}" if a["Z"] is not None else f"{'-':>7}"
        cap = f"{m['at_cap']:>8.2f}" if m.get("at_cap") is not None else f"{'-':>8}"
        kl = f"{m['kl_last']:>9.4f}" if m.get("kl_last") is not None else f"{'-':>9}"
        lr = f"{m['lr_last']:>9.1e}" if m.get("lr_last") is not None else f"{'-':>9}"
        ent = f"{m['entropy_end']:>9.3f}" if m.get("entropy_end") is not None else f"{'-':>9}"
        sg = f"{m['S_growth']:>6.0f}" if m.get("S_growth") is not None else f"{'-':>6}"
        sps = f"{m['sps']:>6.0f}" if m.get("sps") is not None else f"{'-':>6}"
        print(f"{a['arm']:<13}{row}{s11}{y}{PREDICTED[a['arm']]:>9.3f}{z}{cap}{kl}{lr}{ent}{sg}{sps}")

    out = {"arms": arms, "newest": newest}
    by = {a["arm"]: a for a in scored}
    if "c0-e0-f0-g0" in by and "control" in by:
        d = by["c0-e0-f0-g0"]["Y"] - by["control"]["Y"]
        out["W_minus_control"] = d
        print(f"\nW − đối chứng ở seed 2: {d:+.3f} (dự đoán +0,13 ± 0,10; ở seed 1: +0,185)")
    if all(a in by for a in DESIGN):
        print("\nHiệu ứng chính trên Y (dự đoán trong ngoặc; sai số chỉ tính phần nhị thức):")
        eff = {}
        for f in "cefg":
            hi = [by[a]["Y"] for a in DESIGN if levels(a)[f]]
            lo = [by[a]["Y"] for a in DESIGN if not levels(a)[f]]
            p = sum(hi + lo) / 8
            se = math.sqrt(p * (1 - p) / 200 * (1 / 4 + 1 / 4))
            eff[f.upper()] = sum(hi) / 4 - sum(lo) / 4
            print(f"  {f.upper()} ({FACTORS[f]}): {eff[f.upper()]:+.3f} ± {se:.3f}   (dự đoán {PRED_EFFECTS[f.upper()]:+.3f})")
        for pair in ("ce", "cf", "cg"):
            sign = [(1 if levels(a)[pair[0]] else -1) * (1 if levels(a)[pair[1]] else -1) for a in DESIGN]
            val = sum(s * by[a]["Y"] for s, a in zip(sign, DESIGN)) / 4
            alias = {"ce": "FG", "cf": "EG", "cg": "EF"}[pair]
            eff[pair.upper() + "/" + alias] = val
            print(f"  cặp {pair.upper()}/{alias}: {val:+.3f}")
        out["effects_Y"] = eff
        if args.effects_plot:
            effects_plot(eff, args.effects_plot)
            print("đã ghi", args.effects_plot)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    if args.plot:
        plot(arms, args.plot, newest)
        print("đã ghi", args.plot)


if __name__ == "__main__":
    main()
