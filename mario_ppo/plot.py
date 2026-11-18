"""Vẽ đồ thị quá trình huấn luyện từ progress.csv và evals.csv (công cụ thêm của nhóm, cần matplotlib).

    python -m mario_ppo.plot runs/1-1/<run>                    # ghi runs/1-1/<run>/curves.png
    python -m mario_ppo.plot runs/a runs/b --out so-sanh.png   # chồng nhiều lần chạy để so sánh

Mỗi ô là một đại lượng trên trục riêng của nó, trục ngang là số bước tác tử.
Khi có nhiều điểm, đường mờ là giá trị từng lần cập nhật, đường đậm là trung
bình trượt.
"""

import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.ticker import FuncFormatter

# Bảng màu phân loại theo thứ tự cố định; lần chạy thứ i luôn nhận màu thứ i.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

PANELS = [
    ("ep_return", "Lợi tức mỗi lượt", "trung bình 100 lượt huấn luyện gần nhất"),
    ("x_pos", "Vị trí x cuối lượt", "điểm ảnh, trung bình 100 lượt gần nhất"),
    ("flag_rate", "Tỉ lệ chạm cờ", "đường: khi huấn luyện; chấm: các lần đánh giá"),
    ("ep_length", "Độ dài lượt", "bước tác tử"),
    ("entropy", "Entropy của chính sách", "cao = còn thăm dò, thấp = đã quyết đoán"),
    ("approx_kl", "approxKL", "mức thay đổi chính sách mỗi lần cập nhật"),
    ("clipfrac", "Tỉ lệ cắt", "phần mẫu có tỉ số chính sách vượt ngưỡng cắt"),
    ("explained_var", "Phương sai được giải thích", "critic dự đoán lợi tức tốt đến đâu (tối đa 1)"),
]


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _column(rows, key):
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, ValueError):
            values.append(math.nan)
    return values


def _rolling(values, window):
    """Trung bình trượt bỏ qua giá trị thiếu."""
    out = []
    for index in range(len(values)):
        chunk = [v for v in values[max(0, index - window + 1) : index + 1] if not math.isnan(v)]
        out.append(sum(chunk) / len(chunk) if chunk else math.nan)
    return out


def _steps(value, _position=None):
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:g}M".replace(".", ",")
    if abs(value) >= 1_000:
        return f"{value / 1_000:g}k".replace(".", ",")
    return f"{value:g}"


def _number(value):
    if math.isnan(value):
        return "chưa có"
    if abs(value) >= 100:
        return f"{value:,.0f}".replace(",", ".")
    return f"{value:.3g}".replace(".", ",")


def load_run(run_dir):
    """Đọc một thư mục lần chạy: progress.csv, evals.csv và config.yaml."""
    config = {}
    config_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    return {
        "name": os.path.basename(os.path.normpath(run_dir)),
        "progress": _read_csv(os.path.join(run_dir, "progress.csv")),
        "evals": _read_csv(os.path.join(run_dir, "evals.csv")),
        "config": config,
    }


def plot_runs(run_dirs, out_path):
    """Vẽ các lần chạy trong run_dirs vào một tệp ảnh; trả đường dẫn tệp đã ghi."""
    runs = [load_run(run_dir) for run_dir in run_dirs][: len(SERIES)]
    runs = [run for run in runs if run["progress"]]
    if not runs:
        raise SystemExit("Không có progress.csv nào có dữ liệu để vẽ.")
    single = len(runs) == 1

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    figure, axes = plt.subplots(2, 4, figsize=(16, 7.6), dpi=130, facecolor=SURFACE)
    figure.subplots_adjust(left=0.045, right=0.965, top=0.84, bottom=0.08, wspace=0.28, hspace=0.5)

    for axis, (key, title, note) in zip(axes.flat, PANELS):
        axis.set_facecolor(SURFACE)
        for index, run in enumerate(runs):
            color = SERIES[index]
            steps = _column(run["progress"], "global_step")
            values = _column(run["progress"], key)
            if len(values) > 50:
                axis.plot(steps, values, color=color, linewidth=1, alpha=0.3)
                smooth = _rolling(values, max(5, len(values) // 50))
                axis.plot(steps, smooth, color=color, linewidth=2, label=run["name"])
                last = smooth[-1]
            else:
                axis.plot(steps, values, color=color, linewidth=2, label=run["name"])
                last = values[-1]
            if single:
                axis.annotate(
                    _number(last), (steps[-1], last), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_SECONDARY, annotation_clip=False,
                )
            if key == "flag_rate" and run["evals"]:
                axis.plot(
                    _column(run["evals"], "global_step"), _column(run["evals"], "flag_rate"),
                    linestyle="none", marker="o", markersize=7, color=SERIES[1] if single else color,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label="đánh giá" if single else None,
                    zorder=3,
                )

        if key == "flag_rate":
            axis.axhline(0.8, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            axis.text(0.01, 0.8, " ngưỡng 0,8 của đề", transform=axis.get_yaxis_transform(),
                      va="bottom", fontsize=8, color=MUTED)
            axis.set_ylim(-0.03, 1.03)
            if single and runs[0]["evals"]:
                handles, labels = axis.get_legend_handles_labels()
                axis.legend(handles, ["huấn luyện", "đánh giá"], loc="center left", frameon=False,
                            fontsize=8, labelcolor=INK_SECONDARY)
        if key == "approx_kl":
            target = (runs[0]["config"].get("algo") or {}).get("target_kl")
            if target:
                axis.axhline(target, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
                axis.text(0.01, target, " ngưỡng dừng sớm", transform=axis.get_yaxis_transform(),
                          va="top", fontsize=8, color=MUTED)

        axis.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=20)
        axis.text(0, 1.03, note, transform=axis.transAxes, fontsize=8, color=MUTED, va="bottom")
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED, labelsize=8, length=0)
        axis.xaxis.set_major_formatter(FuncFormatter(_steps))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
        axis.margins(x=0.02)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(AXIS)

    for axis in axes[1]:
        axis.set_xlabel("bước tác tử", fontsize=8, color=MUTED)

    first = runs[0]
    last_step = _column(first["progress"], "global_step")[-1]
    if single:
        level = first["config"].get("level", "?")
        heading = f"Màn {level} · {first['name']} · {_number(last_step)} bước"
        latest = first["progress"][-1]
        summary = (
            f"Hiện tại: tỉ lệ chạm cờ {_number(float(latest.get('flag_rate') or 'nan'))} · "
            f"x cuối lượt {_number(float(latest.get('x_pos') or 'nan'))} · "
            f"tốc độ {_number(float(latest.get('sps') or 'nan'))} bước/giây"
        )
    else:
        heading = "So sánh các lần chạy"
        summary = "Mỗi màu là một lần chạy; màu gắn với lần chạy theo thứ tự truyền vào."
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.975), ncol=min(len(runs), 4),
                      frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    figure.text(0.045, 0.955, heading, fontsize=15, fontweight="bold", color=INK, va="center")
    figure.text(0.045, 0.915, summary, fontsize=10, color=INK_SECONDARY, va="center")

    temp_path = f"{out_path}.tmp.png"
    figure.savefig(temp_path, facecolor=SURFACE)
    plt.close(figure)
    os.replace(temp_path, out_path)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vẽ đồ thị huấn luyện từ thư mục các lần chạy.")
    parser.add_argument("run_dirs", nargs="+", help="một hoặc nhiều thư mục lần chạy")
    parser.add_argument("--out", default=None, help="tệp ảnh đầu ra; mặc định <run_dir>/curves.png")
    args = parser.parse_args(argv)
    out_path = args.out or os.path.join(args.run_dirs[0], "curves.png")
    print(f"Đã ghi {plot_runs(args.run_dirs, out_path)}")


if __name__ == "__main__":
    main()
