"""Ba chỉ số ở bức tường lò xo của 2-1, tính từ trace.npz của experiments/value_trace.py (không chạy trò chơi).

Định nghĩa giống experiments/clock_swap.py và bảng 39/10 trong docs/extension-directions.md:
  - "tới tường": x lớn nhất của lượt >= 2960;
  - "ở tường": bước có x trong [2960, 3060];
  - "qua tường": x lớn nhất > 3130;
  - "qua sớm": lượt qua tường mà số bước ở tường đếm tới lúc x lớn nhất vượt 3130 không quá 200.

    python experiments/wall_metrics.py <thư mục có trace.npz> [<thư mục> ...] [--json out.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

WALL_LO, WALL_HI, PASSED, EARLY = 2960, 3060, 3130, 200


def wall_metrics(trace_path):
    d = np.load(trace_path)
    meta = json.loads(str(d["meta"]))
    ep, x = d["ep"], d["x"].astype(int)
    rows = []
    for i in range(len(d["ep_x_end"])):
        xs = x[ep == i]
        best, stood, pass_stood = 0, 0, None
        for xi in xs:  # cùng thứ tự cập nhật với clock_swap.py
            best = max(best, int(xi))
            if WALL_LO <= xi <= WALL_HI:
                stood += 1
            if best > PASSED and pass_stood is None:
                pass_stood = stood
        best = max(best, int(d["ep_x_end"][i]))
        if best > PASSED and pass_stood is None:
            pass_stood = stood
        rows.append({"flag": str(d["ep_cause"][i]) == "flag", "reached": best >= WALL_LO,
                     "passed": best > PASSED, "stood": stood, "stood_at_pass": pass_stood})
    reached = [r for r in rows if r["reached"]]
    passed = [r for r in rows if r["passed"]]
    early = [r for r in passed if r["stood_at_pass"] <= EARLY]
    stood_total = sum(r["stood"] for r in reached)
    return {"checkpoint": meta.get("checkpoint"), "global_step": meta.get("global_step"), "seed": meta.get("seed"),
            "episodes": len(rows), "flags": sum(r["flag"] for r in rows), "reached": len(reached),
            "passed": len(passed), "early_pass": len(early),
            "early_pass_rate": f"{len(early)}/{len(reached)}" if reached else "không ước lượng được",
            "passes_per_1000_wall_steps": round(1000 * len(passed) / stood_total, 2) if stood_total else None,
            "stood_at_pass": sorted(r["stood_at_pass"] for r in passed)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    out = []
    print(f"{'trace':78s} {'seed':>4} {'cờ':>4} {'tới':>4} {'qua':>4} {'qua sớm/tới':>12} {'qua/1000 bước':>13}")
    for folder in args.dirs:
        r = wall_metrics(Path(folder) / "trace.npz")
        r["dir"] = folder
        out.append(r)
        print(f"{folder:78s} {r['seed']:>4} {r['flags']:>4} {r['reached']:>4} {r['passed']:>4} "
              f"{r['early_pass_rate']:>12} {str(r['passes_per_1000_wall_steps']):>13}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
