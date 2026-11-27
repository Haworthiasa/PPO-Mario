"""Vẽ tỉ lệ chạm cờ lúc huấn luyện của MỘT màn qua mọi lượt chạy có huấn luyện màn đó (chỉ đọc tệp, không huấn luyện).

    python experiments/level_curves.py 1-1            # ghi docs/reports/figures/1-1-training.png
    python experiments/level_curves.py 2-1 --out x.png

Hình có hai khung: theo tổng số bước của lượt chạy, và theo số bước riêng trên màn đó. Số liệu lấy từ `progress.csv`
(tỉ lệ chạm cờ của 100 lượt huấn luyện gần nhất), nên đây là số lúc huấn luyện, không phải số chấm bằng evaluate.py.
Thêm lượt chạy mới: thêm một dòng vào RUNS.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# share: "all" (lượt một màn), "fixed" (mỗi màn giữ nguyên 1/số màn), "alloc" (1/số màn tới khi allocation.csv bắt đầu,
# sau đó cộng dồn cột steps_<màn>), "resumed" (nối tiếp một checkpoint: `before` là số bước riêng đã có trước `start`).
RUNS = [
    dict(path="legacy/runs/trial-100k", label="Bản tự viết, một màn", levels=["1-1"], share="all"),
    dict(path="runs/1-1/run-1", label="Một màn (bản nộp 1-1)", levels=["1-1"], share="all"),
    dict(path="runs/4-1/from-scratch", label="Một màn (bản nộp 4-1)", levels=["4-1"], share="all"),
    dict(path="runs/4-1/transfer", label="Chuyển giao từ 1-1", levels=["4-1"], share="resumed",
         start=1200128, before={"4-1": 0}, after=1),
    dict(path="runs/2-3/from-scratch", label="Một màn, 1,2M", levels=["2-3"], share="all"),
    dict(path="runs/2-3/from-scratch-3m", label="Một màn, 3M (bản nộp 2-3)", levels=["2-3"], share="all"),
    dict(path="runs/1-1+4-1/from-scratch", label="Một mạng 2 màn", levels=["1-1", "4-1"], share="fixed"),
    dict(path="runs/multi-level/restart-test", label="Mạng 2 màn thêm 6 màn mới (thử)",
         levels=["1-1", "4-1", "2-1", "3-1", "3-2", "5-1", "6-1", "8-1"], share="resumed",
         start=2904064, before={"1-1": 1452032, "4-1": 1452032}, after=8),
    dict(path="runs/8-levels/from-scratch-10m", label="Một mạng 8 màn",
         levels=["1-1", "2-1", "3-1", "3-2", "4-1", "5-1", "5-2", "6-1"], share="alloc"),
    dict(path="runs/2-1/wall-cut-2000", label="Nối từ 6,30M, chỉ 2-1, cắt 2000 bước", levels=["2-1"], share="resumed",
         start=6303744, before={"2-1": 1034903}, after=1),
    dict(path="runs/2-1/wall-cut-200", label="Nối từ 6,30M, chỉ 2-1, cắt 200 bước", levels=["2-1"], share="resumed",
         start=6303744, before={"2-1": 1034903}, after=1),
    dict(path="runs/2-1/wall-cut-2000-seed2", label="Như trên, cắt 2000 bước, seed 2", levels=["2-1"], share="resumed",
         start=6303744, before={"2-1": 1034903}, after=1),
    dict(path="runs/2-1/wall-cut-200-seed2", label="Như trên, cắt 200 bước, seed 2", levels=["2-1"], share="resumed",
         start=6303744, before={"2-1": 1034903}, after=1),
    dict(path="runs/2-1/gamma-0.97-cut-200", label="Như trên, cắt 200 bước, γ = 0,97", levels=["2-1"],
         share="resumed", start=6303744, before={"2-1": 1034903}, after=1),
]
MIN_EPISODES = 20  # bỏ các dòng mà cửa sổ của màn có ít hơn chừng này lượt
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DASHES = [(0, (3, 2)), "solid", (0, (6, 3)), (0, (1, 1.5)), (0, (8, 2, 1.5, 2)), (0, (4, 1)), (0, (2, 1)), "solid"]


def curve(run, level):
    """[(tổng bước, bước riêng trên màn, tỉ lệ chạm cờ)] của một lượt chạy; [] nếu chưa có số nào."""
    rows = list(csv.DictReader(open(ROOT / run["path"] / "progress.csv", encoding="utf-8")))
    column = "flag_rate" if len(run["levels"]) == 1 else f"flag_rate_{level}"
    n = len(run["levels"])
    own_at, first_alloc = {}, None
    if run["share"] == "alloc":
        alloc = list(csv.DictReader(open(ROOT / run["path"] / "allocation.csv", encoding="utf-8")))
        first_alloc = int(alloc[0]["global_step"])
        total = (first_alloc - int(alloc[0][f"steps_{level}"]) * n) // n
        for a in alloc:
            total += int(a[f"steps_{level}"])
            own_at[int(a["global_step"])] = total
    out, own = [], 0
    for r in rows:
        g, value = int(r["global_step"]), r.get(column, "")
        if run["share"] == "all":
            own = g
        elif run["share"] == "fixed":
            own = g // n
        elif run["share"] == "resumed":
            own = run["before"].get(level, 0) + (g - run["start"]) // run["after"]
        else:
            own = own_at.get(g, own if g >= first_alloc else g // n)
        seen = r.get("episodes" if len(run["levels"]) == 1 else f"episodes_{level}", "")
        if seen not in ("", "nan") and float(seen) < MIN_EPISODES:
            continue  # cửa sổ 100 lượt vừa bị đặt lại (đầu lượt chạy hoặc ngay sau khi nối tiếp): số chưa có nghĩa
        if value not in ("", "nan"):
            out.append((g / 1e6, own / 1e6, 100 * float(value)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("level")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    runs = [r for r in RUNS if args.level in r["levels"] and (ROOT / r["path"] / "progress.csv").exists()]
    if not runs:
        raise SystemExit(f"không có lượt chạy nào huấn luyện màn {args.level}")
    out = Path(args.out) if args.out else ROOT / "docs" / "reports" / "figures" / f"{args.level}-training.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.2), dpi=150)
    titles = ("Theo tổng số bước của lượt chạy (triệu bước)", f"Theo số bước riêng trên màn {args.level} (triệu bước)")
    for ax, column, title in zip(axes, (0, 1), titles):
        for i, run in enumerate(runs):
            points = curve(run, args.level)
            if not points:
                continue
            ax.plot([q[column] for q in points], [q[2] for q in points], color=COLORS[i % len(COLORS)],
                    linestyle=DASHES[i % len(DASHES)], linewidth=1.8, label=run["label"])
            print(f"{run['path']}: {len(points)} điểm, cuối ở {points[-1][0]:.3f}M tổng / {points[-1][1]:.3f}M riêng, "
                  f"chạm cờ {points[-1][2]:.0f}%")
        ax.axhline(80, color="#52514e", linestyle=(0, (1, 3)), linewidth=1, label="Ngưỡng vượt màn 80%")
        ax.set_ylim(0, 100)
        ax.set_xlim(left=0)
        ax.set_yticks(range(0, 101, 20))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors="#898781", length=0)
        ax.set_title(title, loc="left", fontsize=10, color="#52514e")
    # Chú thích chung đặt dưới tiêu đề hình; chiều cao phần đầu tăng theo số dòng chú thích để không đè lên ô đầu.
    handles, labels = axes[0].get_legend_handles_labels()
    rows = -(-len(labels) // 2)
    head = 0.45 + 0.24 * rows  # inch
    fig.set_size_inches(8, 6.6 + head)
    top = 1 - head / (6.6 + head)
    fig.suptitle(f"Màn {args.level}: tỉ lệ chạm cờ lúc huấn luyện (100 lượt gần nhất)", x=0.07, y=0.995, ha="left",
                 va="top", fontsize=11)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.4 / (6.6 + head)), ncol=2,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out, facecolor="white")
    print("đã ghi", out)


if __name__ == "__main__":
    main()
