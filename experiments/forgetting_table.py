"""Bảng độ quên: các nhánh chỉ học 2-1 (nối tiếp bước 6.303.744 của lượt 8 màn) còn chơi được bảy màn kia không.

Đọc các tệp JSON do mario_ppo.evaluate ghi (100 lượt, seed 7) ở runs/2-1/forgetting/<nhãn>/<màn>.json và in bảng
chạm cờ /100 và x cuối trung bình theo màn, kèm chênh lệch so với điểm xuất phát (nhãn base-6303744).

Chạy các lần chấm (trong WSL, từ thư mục gốc của kho), ví dụ cho một checkpoint:
    OMP_NUM_THREADS=2 python -m mario_ppo.evaluate --ckpt <ckpt> --level 1-1 --episodes 100 --seed 7 \
        --out runs/2-1/forgetting/<nhãn>/1-1.json
Rồi:
    python experiments/forgetting_table.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "runs" / "2-1" / "forgetting"
LEVELS = ["1-1", "3-1", "3-2", "4-1", "5-1", "5-2", "6-1"]
LABELS = [("base-6303744", "Điểm xuất phát: mạng 8 màn, bước 6.303.744"),
          ("gamma-0.90-cut-200-s1", "γ 0,9, cắt 200, seed 1 (wall-cut-200)"),
          ("gamma-0.90-cut-200-s2", "γ 0,9, cắt 200, seed 2 (wall-cut-200-seed2)"),
          ("gamma-0.97-cut-200-s1", "γ 0,97, cắt 200, seed 1 (gamma-0.97-cut-200)"),
          ("gamma-0.97-cut-200-s2", "γ 0,97, cắt 200, seed 2 (gamma-0.97-cut-200-seed2)")]


def load(label, level):
    f = ROOT / label / f"{level}.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    return round(r["flag_rate"] * r["episodes"]), r["mean_x_pos"]


def main():
    base = {lv: load("base-6303744", lv) for lv in LEVELS}
    print("Chạm cờ /100 (seed 7); trong ngoặc: chênh so với điểm xuất phát")
    print(f"{'nhánh':48s} " + " ".join(f"{lv:>9}" for lv in LEVELS) + f" {'trung bình':>12}")
    for label, name in LABELS:
        cells, flags, base_same = [], [], []
        for lv in LEVELS:
            r = load(label, lv)
            if r is None:
                cells.append(f"{'–':>9}")
                continue
            flags.append(r[0])
            if base[lv]:
                base_same.append(base[lv][0])  # trung bình của điểm xuất phát trên ĐÚNG các màn đã chấm
            d = f"({r[0] - base[lv][0]:+d})" if base[lv] and label != "base-6303744" else ""
            cells.append(f"{r[0]:>3}{d:>6}")
        mean = sum(flags) / len(flags) if flags else float("nan")
        bmean = sum(base_same) / len(base_same) if base_same else float("nan")
        part = "" if len(flags) == len(LEVELS) else f" [{len(flags)}/{len(LEVELS)} màn]"
        tail = f"{mean:6.1f}" + (f" ({mean - bmean:+.1f})" if label != "base-6303744" and flags else "") + part
        print(f"{name:48s} " + " ".join(cells) + f" {tail:>12}")
    print("\nx cuối trung bình (điểm ảnh)")
    for label, name in LABELS:
        cells = []
        for lv in LEVELS:
            r = load(label, lv)
            cells.append(f"{r[1]:9.0f}" if r else f"{'–':>9}")
        print(f"{name:48s} " + " ".join(cells))


if __name__ == "__main__":
    main()
