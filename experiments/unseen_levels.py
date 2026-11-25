"""Chẩn đoán (không huấn luyện): một checkpoint chơi các màn nó CHƯA TỪNG học tới đâu, so với các mạng khác.

    python experiments/unseen_levels.py --ckpt <ckpt> --out-dir <thư mục> --jobs 2
    python experiments/unseen_levels.py --ckpt <ckpt> --out-dir <thư mục> --tables-only    # chỉ dựng lại bảng

Bước 1: với mỗi màn trong 32 màn mà checkpoint chưa học (danh sách màn đã học đọc từ cấu hình trong checkpoint), gọi
script chấm chính thức `python -m mario_ppo.evaluate` với --episodes và --seed cho trước (mặc định 20 lượt, seed 2026,
đúng như lần chấm mạng 2 màn ở runs/1-1+4-1/from-scratch/eval/). Màn nào đã có tệp JSON trong --out-dir thì bỏ qua.
Bước 2: đọc các tệp JSON đó cùng các tệp đã có của mạng 2 màn và ba mạng chuyên một màn, dựng mỗi màn một bảng.
Mạng nào đã học màn đang xét thì dòng của nó được ghi rõ "đã học màn này"; ô không có tệp ghi "chưa đo".

Tệp ghi ra trong --out-dir:
    <màn>.json            kết quả của evaluate.py cho checkpoint này
    unseen-levels.csv     mỗi dòng một cặp (màn, mạng)
    unseen-levels.md      mỗi màn một bảng, kèm một bảng tổng
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALL_LEVELS = [f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)]
# Các mạng đã được chấm từ trước (20 lượt, seed 2026). `other`: tệp trên màn khác; `own`: tệp trên màn nó đã học.
OTHER_NETS = [
    {"name": "mạng 2 màn (1-1 và 4-1), bước 2.904.064", "short": "2 màn", "trained": ["1-1", "4-1"],
     "other": "runs/1-1+4-1/from-scratch/eval/{level}/20-seed2026_002904064.json",
     "own": "runs/1-1+4-1/from-scratch/eval/{level}/20-seed2026_002904064.json"},
    {"name": "mạng chuyên 1-1, bước 1.200.128", "short": "chuyên 1-1", "trained": ["1-1"],
     "other": "runs/1-1/run-1/eval/eval-on-{level}/20-seed2026_001200128.json",
     "own": "runs/1-1/run-1/eval/20-seed2026_001200128.json"},
    {"name": "mạng chuyên 4-1, bước 1.200.128", "short": "chuyên 4-1", "trained": ["4-1"],
     "other": "runs/4-1/from-scratch/eval/eval-on-{level}/20-seed2026_001200128.json",
     "own": "runs/4-1/from-scratch/eval/20-seed2026_001200128.json"},
    {"name": "mạng chuyên 2-3, bước 2.801.664", "short": "chuyên 2-3", "trained": ["2-3"],
     "other": "runs/2-3/from-scratch-3m/eval/eval-on-{level}/20-seed2026_002801664.json",
     "own": "runs/2-3/from-scratch-3m/eval/20-seed2026_002801664.json"},
]


def trained_levels(ckpt):
    import torch
    cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
    return list(cfg.get("levels") or str(cfg["level"]).split("+"))


def evaluate_missing(ckpt, levels, out, episodes, seed, jobs):
    def run(level):
        target = out / f"{level}.json"
        if target.exists():
            return level, "đã có"
        cmd = [sys.executable, "-m", "mario_ppo.evaluate", "--ckpt", str(ckpt), "--level", level,
               "--episodes", str(episodes), "--seed", str(seed), "--out", str(target)]
        done = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"), capture_output=True, text=True)
        lines = (done.stdout.strip() or done.stderr.strip()).splitlines()
        return level, lines[-1] if lines else f"mã thoát {done.returncode}"

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for level, message in pool.map(run, levels):
            print(f"{level}: {message}", flush=True)


def read_result(path, episodes, seed):
    """Trả dict số liệu của một tệp evaluate.py, hoặc None nếu không có tệp. Sai số lượt hay seed thì báo lỗi."""
    path = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    if d["episodes"] != episodes or d["seed"] != seed:
        raise ValueError(f"{path}: {d['episodes']} lượt seed {d['seed']}, cần {episodes} lượt seed {seed}")
    xs = d["x_pos"]
    return {"mean_x": sum(xs) / len(xs), "max_x": max(xs), "flags": round(d["flag_rate"] * len(xs)),
            "episodes": len(xs), "mean_steps": sum(d["steps"]) / len(d["steps"]), "file": str(path)}


def flag_positions():
    """x của cờ từng màn theo runs/level-lengths/level-lengths.csv; màn không đo được cờ thì không có trong dict."""
    with open(ROOT / "runs/level-lengths/level-lengths.csv", encoding="utf-8") as f:
        return {r["level"]: int(r["x"]) for r in csv.DictReader(f) if r["flag_reached"] == "1"}


def build_tables(ckpt, out, episodes, seed):
    trained = trained_levels(ckpt)
    step = int(Path(ckpt).stem.split("_")[-1])
    this_name = f"mạng {len(trained)} màn, bước " + f"{step:,}".replace(",", ".")
    flags = flag_positions()
    unseen = [lv for lv in ALL_LEVELS if lv not in trained]
    rows = []
    for level in unseen:
        r = read_result(out / f"{level}.json", episodes, seed)
        rows.append({"level": level, "net": this_name, "short": f"{len(trained)} màn", "trained_here": False, **(r or {})})
        for net in OTHER_NETS:
            own = level in net["trained"]
            r = read_result((net["own"] if own else net["other"]).format(level=level), episodes, seed)
            rows.append({"level": level, "net": net["name"], "short": net["short"], "trained_here": own, **(r or {})})

    with open(out / "unseen-levels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "flag_x", "network", "trained_on_this_level", "mean_x", "max_x", "flags", "episodes",
                    "mean_steps", "file"])
        for r in rows:
            w.writerow([r["level"], flags.get(r["level"], ""), r["net"], int(r["trained_here"])] +
                       ([f"{r['mean_x']:.1f}", r["max_x"], r["flags"], r["episodes"], f"{r['mean_steps']:.0f}", r["file"]]
                        if "mean_x" in r else ["", "", "", "", "", ""]))

    def pct(x, level):
        return f"{100 * x / flags[level]:.0f}%" if level in flags else "–"

    md = [f"# Các màn chưa học: {this_name} so với các mạng khác", "",
          f"Chấm bằng `mario_ppo.evaluate`, {episodes} lượt, seed {seed}, lấy mẫu hành động. Màn mạng {len(trained)} màn "
          f"đã học ({', '.join(trained)}) không có ở đây.",
          "\"% màn\" = x chia cho x của cờ theo `runs/level-lengths/level-lengths.csv`; dấu – là màn không đo được cờ "
          "(kết thúc bằng ống hoặc mê cung).", ""]
    shorts = [f"{len(trained)} màn"] + [n["short"] for n in OTHER_NETS]
    md += ["## Bảng tổng: x trung bình (trong ngoặc là x xa nhất)", "",
           "| Màn | x cờ | " + " | ".join(shorts) + " |", "|---|---|" + "---|" * len(shorts)]
    for level in unseen:
        cells = []
        for r in (r for r in rows if r["level"] == level):
            if "mean_x" not in r:
                cells.append("chưa đo")
            else:
                text = f"{r['mean_x']:.0f} ({r['max_x']})"
                if r["flags"]:
                    text += f", {r['flags']} cờ"
                cells.append(f"*đã học:* {text}" if r["trained_here"] else text)
        md.append(f"| {level} | {flags.get(level, '–')} | " + " | ".join(cells) + " |")
    md.append("")

    md += ["## Mỗi mạng đi được bao nhiêu phần màn (chỉ các màn đo được x của cờ, chỉ các màn mạng đó chưa học)", "",
           "| Mạng | Số màn | Lượt chạm cờ | Trung bình của (x trung bình / x cờ) | Trung bình của (x xa nhất / x cờ) | "
           "Lớn nhất của (x xa nhất / x cờ) |", "|---|---|---|---|---|---|"]
    for short, name in [(f"{len(trained)} màn", this_name)] + [(n["short"], n["name"]) for n in OTHER_NETS]:
        rs = [r for r in rows if r["short"] == short and "mean_x" in r and not r["trained_here"] and r["level"] in flags]
        if rs:
            md.append(f"| {name} | {len(rs)} | {sum(r['flags'] for r in rs)}/{sum(r['episodes'] for r in rs)} | "
                      f"{100 * sum(r['mean_x'] / flags[r['level']] for r in rs) / len(rs):.1f}% | "
                      f"{100 * sum(r['max_x'] / flags[r['level']] for r in rs) / len(rs):.1f}% | "
                      f"{100 * max(r['max_x'] / flags[r['level']] for r in rs):.0f}% |")
    md.append("")

    # So từng cặp trên những màn mà cả hai mạng đều có số và đều chưa học
    first = f"{len(trained)} màn"
    md += [f"## So từng cặp: mạng {first} với từng mạng khác, trên các màn cả hai đều chưa học và đều đã chấm", "",
           "| Mạng so sánh | Số màn chung | Mạng " + first + " xa hơn (x trung bình) | Hòa (chênh dưới 50) | "
           "Trung bình của x trung bình: " + first + " / mạng kia | Trung bình của x xa nhất: " + first + " / mạng kia |",
           "|---|---|---|---|---|---|"]
    for net in OTHER_NETS:
        pairs = []
        for level in unseen:
            mine = next(r for r in rows if r["level"] == level and r["short"] == first)
            other = next(r for r in rows if r["level"] == level and r["short"] == net["short"])
            if "mean_x" in mine and "mean_x" in other and not other["trained_here"]:
                pairs.append((mine, other))
        if not pairs:
            continue
        better = sum(m["mean_x"] - o["mean_x"] >= 50 for m, o in pairs)
        tie = sum(abs(m["mean_x"] - o["mean_x"]) < 50 for m, o in pairs)
        md.append(f"| {net['name']} | {len(pairs)} | {better} | {tie} | "
                  f"{sum(m['mean_x'] for m, _ in pairs) / len(pairs):.0f} / {sum(o['mean_x'] for _, o in pairs) / len(pairs):.0f} | "
                  f"{sum(m['max_x'] for m, _ in pairs) / len(pairs):.0f} / {sum(o['max_x'] for _, o in pairs) / len(pairs):.0f} |")
    md.append("")
    for level in unseen:
        md += [f"## Màn {level}" + (f" (cờ ở x {flags[level]})" if level in flags else " (không đo được x của cờ)"), "",
               "| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |",
               "|---|---|---|---|---|---|"]
        for r in (r for r in rows if r["level"] == level):
            note = "đã học màn này, không tính là màn lạ" if r["trained_here"] else "chưa học màn này"
            if "mean_x" not in r:
                md.append(f"| {r['net']} | chưa đo | chưa đo | chưa đo | chưa đo | {note} |")
            else:
                md.append(f"| {r['net']} | {r['mean_x']:.0f} | {r['max_x']} | {pct(r['mean_x'], level)} / "
                          f"{pct(r['max_x'], level)} | {r['flags']}/{r['episodes']} | {note} |")
        md.append("")
    (out / "unseen-levels.md").write_text("\n".join(md), encoding="utf-8")
    print(f"đã ghi {out / 'unseen-levels.md'} và unseen-levels.csv ({len(unseen)} màn chưa học)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--jobs", type=int, default=2, help="số tiến trình chấm chạy song song")
    p.add_argument("--tables-only", action="store_true", help="không chấm, chỉ dựng bảng từ các tệp JSON đã có")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.tables_only:
        unseen = [lv for lv in ALL_LEVELS if lv not in trained_levels(args.ckpt)]
        evaluate_missing(args.ckpt, unseen, out, args.episodes, args.seed, args.jobs)
    build_tables(args.ckpt, out, args.episodes, args.seed)


if __name__ == "__main__":
    main()
