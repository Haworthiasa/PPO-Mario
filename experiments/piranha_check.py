"""Chẩn đoán (không huấn luyện): cây ăn thịt chui lên từ cống có làm Mario chết không, và khi nào nó chui lên.

    python experiments/piranha_check.py play --ckpt <ckpt> --levels 2-1 4-1 --episodes 20 --out-dir <thư mục>
    python experiments/piranha_check.py poke --ckpt <ckpt> --levels 2-1 --tries 15 --out-dir <thư mục>

Mọi số đo đọc từ RAM của trình giả lập sau mỗi bước tác tử (4 khung hình). Mạng vẫn chỉ nhận khung hình đã tiền
xử lý; không con số nào ở đây được dùng để huấn luyện, và script không ghi gì vào RAM.

play  Mạng tự chơi. Ghi: mỗi lượt chết vì địch thì địch nào ở gần hộp va chạm của Mario nhất; số bước mà hình vẽ
      của Mario chồng lên hình vẽ của cây nhưng hai hộp va chạm không chạm nhau; khoảng cách ngang |dx| giữa
      Mario và cây lúc cây bắt đầu trồi lên.
poke  Thử trực tiếp. Mạng lái Mario tới gần một cái cống có cây, rồi script cầm lái:
        1. ép (--press-steps, 0 = bỏ): giữ nút "phải" để Mario đứng sát cống, đếm số lần cây trồi lên;
        2. vào vị trí: đi tới chỗ cách cây một khoảng định trước (36 tới 56 px);
        3. chờ: đứng yên tới khi cây đã nhô lên một đoạn định trước;
        4. nhảy: giữ "phải + nhảy" (hoặc thêm nút chạy) 45 bước để rơi vào cây, xem Mario có chết không.

Địa chỉ RAM (theo bản dịch ngược mã Super Mario Bros.): 0x0E trạng thái Mario (0x0B = đang chết); 0x0F+i ô địch
i đang dùng; 0x16+i loại địch (0x0D = cây ăn thịt); 0x6D/0x86 trang và x của Mario; 0x6E+i/0x87+i trang và x của
địch; 0xCE, 0xCF+i toạ độ y; 0x417+i và 0x434+i độ cao trên cùng và dưới cùng của cây; 0x4AC..0x4AF hộp va chạm
của Mario (trái, trên, phải, dưới, theo toạ độ màn hình); 0x4B0+4i hộp va chạm của địch i.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.death_look import cause_of
from mario_ppo.env import make_env
from mario_ppo.evaluate import NOOP, load_model

PIRANHA = 0x0D
NAMES = {0x00: "rùa xanh", 0x01: "rùa đỏ", 0x02: "bọ mai cứng", 0x03: "rùa đỏ", 0x05: "anh em búa", 0x06: "nấm goomba",
         0x07: "mực", 0x08: "đạn", 0x0A: "cá", 0x0B: "cá", 0x0C: "cầu lửa", 0x0D: "cây ăn thịt",
         0x0E: "rùa xanh có cánh", 0x0F: "rùa đỏ có cánh", 0x10: "rùa xanh có cánh", 0x11: "lakitu", 0x12: "nhím",
         0x14: "cá bay"}
RIGHT, RIGHT_JUMP, LEFT = 1, 2, 6     # chỉ số trong bộ hành động "simple"


def read(env):
    ram = env.unwrapped.ram
    px = 256 * int(ram[0x6D]) + int(ram[0x86])
    s = {"state": int(ram[0x0E]), "px": px, "py": int(ram[0xCE]), "box": [int(v) for v in ram[0x4AC:0x4B0]],
         "small": int(ram[0x754]), "air": int(ram[0x1D]), "enemies": []}
    for i in range(5):
        if not ram[0x0F + i]:
            continue
        s["enemies"].append({"slot": i, "id": int(ram[0x16 + i]), "x": 256 * int(ram[0x6E + i]) + int(ram[0x87 + i]),
                             "y": int(ram[0xCF + i]), "up": int(ram[0x417 + i]), "down": int(ram[0x434 + i]),
                             "box": [int(v) for v in ram[0x4B0 + 4 * i:0x4B4 + 4 * i]]})
    return s


def valid(box):
    return box[0] < box[2] and box[1] < box[3] and box != [255, 255, 255, 255]


def gap(a, b):
    """Khoảng hở giữa hai hộp (px); <= 0 là hai hộp chồng nhau."""
    return max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3])


def sprites_touch(s, e):
    """Hình vẽ (không phải hộp va chạm) của Mario và phần cây nhô khỏi cống có chồng nhau không. Hình Mario suy
    từ hộp va chạm: rộng 16, cao 16 (nhỏ) hoặc 32 (lớn), đáy trùng đáy hộp. Cây: rộng 16, từ y tới miệng cống."""
    if not valid(s["box"]) or e["y"] >= e["down"]:
        return False
    mario = [s["px"], s["box"][3] - (16 if s["small"] else 32), s["px"] + 16, s["box"][3]]
    plant = [e["x"], e["y"], e["x"] + 16, e["down"]]
    return gap(mario, plant) < 0


def killer(s):
    best = None
    for e in s["enemies"]:
        if valid(e["box"]) and valid(s["box"]):
            g = gap(s["box"], e["box"])
            if best is None or g < best[1]:
                best = (e, g)
    return best


def plants(s):
    return [e for e in s["enemies"] if e["id"] == PIRANHA]


def play(model, level, env_cfg, episodes, seed, out, stall=300, max_noop=30, shots=6):
    import imageio.v3 as iio
    env = make_env(level, **dict(env_cfg, eval_mode=True))
    rng = np.random.default_rng(seed)
    noops = [int(n) for n in rng.integers(0, max_noop + 1, size=episodes)]
    torch.manual_seed(seed)
    res = {"level": level, "episodes": episodes, "causes": Counter(), "killers": Counter(), "plant_xs": set(),
           "plant_steps": 0, "hit_steps": 0, "sprite_only_steps": 0, "sprite_only_episodes": 0, "rise_dx": [],
           "down_near_steps": 0, "down_near_rises": 0, "box_offsets": Counter(), "deaths_by_plant": []}
    n_shot = {"death": 0, "touch": 0}
    for i, n in enumerate(noops):
        obs, info = env.reset(seed=seed + i)
        done, steps, best_x, since, prev, touched = False, 0, 0, 0, {}, False
        for _ in range(n):
            obs, _, term, trunc, info = env.step(NOOP)
            done = done or term or trunc
        while not done and steps < 3000 and since < stall:
            action, _, _ = model.act(torch.as_tensor(np.asarray(obs)[None]))
            obs, _, term, trunc, info = env.step(int(action))
            done, steps = term or trunc, steps + 1
            s = read(env)
            since = 0 if s["px"] > best_x else since + 1
            best_x = max(best_x, s["px"])
            now = {}
            for e in plants(s):
                key = e["x"]
                now[key] = e["y"]
                res["plant_xs"].add(key)
                res["plant_steps"] += 1
                dx = abs(e["x"] - s["px"])
                if valid(e["box"]):
                    left = s["px"] - (s["box"][0] - 3) if valid(s["box"]) else None
                    if left is not None:
                        ex = e["x"] - left
                        res["box_offsets"][(e["box"][0] - ex, e["box"][1] - e["y"], e["box"][2] - ex,
                                            e["box"][3] - e["y"])] += 1
                    if valid(s["box"]) and gap(s["box"], e["box"]) <= 0 and not done:
                        res["hit_steps"] += 1
                if key in prev and prev[key] >= e["down"]:      # bước trước cây còn nằm hẳn trong cống
                    rising = e["y"] < e["down"]
                    if rising:
                        res["rise_dx"].append(dx)
                    if dx < 33:
                        res["down_near_steps"] += 1
                        res["down_near_rises"] += int(rising)
                if not done and sprites_touch(s, e) and not (valid(e["box"]) and gap(s["box"], e["box"]) <= 0):
                    res["sprite_only_steps"] += 1
                    if not touched and n_shot["touch"] < shots:
                        iio.imwrite(out / f"{level}-touch-no-death-ep{i}-x{s['px']}.png", env.unwrapped.screen.copy())
                        n_shot["touch"] += 1
                    touched = True
            prev = now
        res["sprite_only_episodes"] += int(touched)
        cause = cause_of(env, info, done) if done else ("stall" if since >= stall else "cap")
        res["causes"][cause] += 1
        if cause == "enemy":
            s = read(env)
            k = killer(s)
            name = "không rõ" if k is None or k[1] > 8 else NAMES.get(k[0]["id"], hex(k[0]["id"]))
            res["killers"][name] += 1
            if k is not None and k[1] <= 8 and k[0]["id"] == PIRANHA:
                res["deaths_by_plant"].append({"episode": i, "x": s["px"], "plant_x": k[0]["x"], "gap": k[1],
                                               "plant_out_px": k[0]["down"] - k[0]["y"]})
                if n_shot["death"] < shots:
                    iio.imwrite(out / f"{level}-death-by-plant-ep{i}-x{s['px']}.png", env.unwrapped.screen.copy())
                    n_shot["death"] += 1
        print(f"{level} ep {i}: {cause}, x {info.get('x_pos')}", flush=True)
    env.close()
    res["plant_xs"] = sorted(res["plant_xs"])
    res["causes"], res["killers"] = dict(res["causes"]), dict(res["killers"])
    res["box_offsets"] = [[list(k), v] for k, v in res["box_offsets"].most_common(3)]
    return res


def play_lines(r):
    lines = [f"=== {r['level']}: {r['episodes']} lượt ===",
             "kết thúc: " + ", ".join(f"{k} {v}" for k, v in sorted(r["causes"].items(), key=lambda kv: -kv[1])),
             "chết vì địch, theo loại địch gần hộp va chạm của Mario nhất: " +
             (", ".join(f"{k} {v}" for k, v in sorted(r["killers"].items(), key=lambda kv: -kv[1])) or "không có"),
             f"số cây ăn thịt khác nhau đã gặp: {len(r['plant_xs'])} (x: {r['plant_xs']})",
             f"số bước có cây trên màn hình: {r['plant_steps']}; bước mà hộp va chạm chồng nhau nhưng Mario không "
             f"chết: {r['hit_steps']}",
             f"bước mà hình Mario chồng hình cây nhưng hộp va chạm không chạm: {r['sprite_only_steps']} "
             f"(trong {r['sprite_only_episodes']} lượt)",
             f"hộp va chạm của cây so với góc trên trái hình cây (trái, trên, phải, dưới): {r['box_offsets']}"]
    if r["rise_dx"]:
        lines.append(f"cây bắt đầu trồi lên {len(r['rise_dx'])} lần; |dx| lúc đó: nhỏ nhất {min(r['rise_dx'])}, "
                     f"trung vị {int(np.median(r['rise_dx']))}")
    lines.append(f"bước cây nằm trong cống khi |dx| < 33: {r['down_near_steps']}; trong đó cây trồi lên ở bước sau: "
                 f"{r['down_near_rises']}")
    for d in r["deaths_by_plant"]:
        lines.append(f"  chết vì cây: lượt {d['episode']}, Mario x {d['x']}, cây x {d['plant_x']}, "
                     f"cây nhô {d['plant_out_px']} px, khoảng hở hộp {d['gap']}")
    return lines


def poke(model, level, env_cfg, tries, seed, out, press_steps=300):
    import imageio.v3 as iio
    env = make_env(level, **dict(env_cfg, eval_mode=True))
    torch.manual_seed(seed)
    rows = []
    for t in range(tries):
        # mỗi lần thử một bộ tham số: đứng cách cây bao xa, nhảy khi cây đã nhô bao nhiêu px, nhảy đi bộ hay nhảy chạy
        trigger, rise, jump = (36, 40, 48, 56)[t % 4], (2, 12, 22)[(t // 4) % 3], (RIGHT_JUMP, 4)[(t // 12) % 2]
        obs, info = env.reset(seed=seed + t)
        row = {"try": t, "trigger_dx": trigger, "rise_px": rise, "jump_action": jump, "plant_x": None,
               "press_steps": 0, "press_rises": 0, "press_dx": None, "wait_steps": 0, "jump_dx": None,
               "jump_steps": 0, "min_gap": None, "phase_at_end": "drive", "end": "alive"}
        phase, target, count, steps, done, was_down = "drive", None, 0, 0, False, True
        while not done and steps < 2300:
            s = read(env)
            p = next((e for e in plants(s) if target is None or abs(e["x"] - target) <= 4), None)
            dx = None if p is None else p["x"] - s["px"]
            if phase == "drive":
                if p is not None and 0 < dx <= 100:
                    phase, target, count = ("press" if press_steps else "position"), p["x"], 0
                    row["plant_x"] = p["x"]
                    was_down = p["y"] >= p["down"]
                    continue
                action = int(model.act(torch.as_tensor(np.asarray(obs)[None]))[0])
            elif p is None:
                row["end"] = "cây biến khỏi bộ nhớ"
                break
            elif phase == "press":
                action, count = RIGHT, count + 1
                if count > 30:                       # 30 bước đầu để Mario kịp đi tới sát cống
                    row["press_steps"] += 1
                    row["press_dx"] = dx
                    down = p["y"] >= p["down"]
                    row["press_rises"] += int(was_down and not down)
                    was_down = down
                if count >= press_steps + 30:
                    phase, count = "position", 0
            elif phase == "position":
                count += 1
                action = RIGHT if dx > trigger + 6 else LEFT if dx < trigger else NOOP
                if (action == NOOP and s["air"] == 0) or count >= 200:
                    phase, count = "wait", 0
                    continue
            elif phase == "wait":
                action, count = NOOP, count + 1
                row["wait_steps"] = count
                if p["down"] - p["y"] >= rise:
                    phase, count = "jump", 0
                    row["jump_dx"] = dx
                    continue
                if count >= 400:
                    row["end"] = "cây không trồi lên"
                    break
            else:
                action, count = jump, count + 1
                row["jump_steps"] = count
                if valid(s["box"]) and valid(p["box"]):
                    g = gap(s["box"], p["box"])
                    row["min_gap"] = g if row["min_gap"] is None else min(row["min_gap"], g)
                if count >= 45:
                    row["end"] = "nhảy trượt"
                    break
            obs, _, term, trunc, info = env.step(action)
            done, steps = term or trunc, steps + 1
            row["phase_at_end"] = phase
        if done:
            s = read(env)
            cause = cause_of(env, info, done)
            k = killer(s) if cause == "enemy" else None
            if k is not None and k[1] <= 8:
                row["end"] = "chết vì " + NAMES.get(k[0]["id"], hex(k[0]["id"]))
                if k[0]["id"] == PIRANHA:
                    row["plant_out_px"], row["death_gap"] = k[0]["down"] - k[0]["y"], k[1]
                    iio.imwrite(out / f"{level}-poke-death-try{t}.png", env.unwrapped.screen.copy())
            else:
                row["end"] = "kết thúc: " + cause
        rows.append(row)
        print(f"{level} try {t}: {row}", flush=True)
    env.close()
    return rows


def poke_lines(level, rows):
    pressed = [r for r in rows if r["press_steps"] >= 100]
    jumped = [r for r in rows if r["jump_dx"] is not None]
    missed = [r for r in jumped if r["end"] == "nhảy trượt"]
    lines = [f"=== {level}: {len(rows)} lần thử ===",
             f"ép sát cống từ 100 bước trở lên: {len(pressed)} lần, tổng {sum(r['press_steps'] for r in pressed)} bước "
             f"({4 * sum(r['press_steps'] for r in pressed)} khung hình); số lần cây trồi lên trong lúc ép: "
             f"{sum(r['press_rises'] for r in pressed)}; dx lúc ép: {sorted({r['press_dx'] for r in pressed})}",
             f"đứng ngoài chờ cây lên rồi nhảy vào: {len(jumped)} lần; số bước chờ cây lên: "
             f"{sorted(r['wait_steps'] for r in jumped)}",
             "kết quả sau khi nhảy: " + ", ".join(f"{k} {v}" for k, v in Counter(r["end"] for r in jumped).most_common()),
             f"các lần nhảy trượt: khoảng hở nhỏ nhất giữa hai hộp va chạm (px, <= 0 là chạm): "
             f"{sorted(r['min_gap'] for r in missed if r['min_gap'] is not None)}",
             "các lần chết vì cây: " + "; ".join(
                 f"lần {r['try']} (đứng cách {r['jump_dx']}, cây nhô {r.get('plant_out_px')} px lúc chết)"
                 for r in jumped if r["end"] == "chết vì cây ăn thịt"),
             "kết quả các lần không tới được bước nhảy: " +
             (", ".join(f"{k} {v}" for k, v in Counter(f"{r['end']} (lúc {r['phase_at_end']})" for r in rows
                                                     if r["jump_dx"] is None).most_common()) or "không có")]
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["play", "poke"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--levels", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--tries", type=int, default=15)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--press-steps", type=int, default=300, help="poke: số bước ép sát cống (0 = bỏ bước ép)")
    p.add_argument("--name", default=None, help="tên tệp kết quả (mặc định là tên chế độ)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, ck = load_model(args.ckpt)
    model.eval()
    lines, data = [], {}
    for level in args.levels:
        if args.mode == "play":
            data[level] = play(model, level, ck["config"]["env"], args.episodes, args.seed, out)
            lines += play_lines(data[level]) + [""]
        else:
            data[level] = poke(model, level, ck["config"]["env"], args.tries, args.seed, out, args.press_steps)
            lines += poke_lines(level, data[level]) + [""]
        (out / f"{args.name or args.mode}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        (out / f"{args.name or args.mode}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
