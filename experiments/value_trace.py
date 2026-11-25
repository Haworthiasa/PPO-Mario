"""Chẩn đoán (không huấn luyện): critic dự đoán gì ở từng bước, nhất là ngay trước khi Mario chết.

    python experiments/value_trace.py --ckpt <ckpt> --level 2-1 --episodes 100 --seed 7 --out-dir <thư mục>
    python experiments/value_trace.py --level 2-1 --out-dir <thư mục> --reuse     # chỉ phân tích lại trace.npz đã có

Script lặp lại đúng các lượt của mario_ppo.evaluate (cùng seed, cùng số bước NOOP, cùng cách lấy mẫu hành động; vòng
lặp theo experiments/death_look.py), nên x cuối từng lượt phải trùng với tệp do evaluate.py hay death_look.py ghi.
Ở mỗi bước tác tử nó ghi thêm những thứ evaluate.py không ghi:
    v        giá trị V mà critic (đầu ra "value" của mạng) dự đoán cho khung hình hiện tại
    x, y     vị trí Mario lúc mạng nhìn khung hình đó (đọc từ info, chỉ để gắn nhãn)
    action   hành động được lấy mẫu; prob: xác suất mạng gán cho hành động đó; probs: xác suất cả 7 hành động
    entropy  entropy của phân phối hành động (0 = chắc chắn một hành động, ln 7 = 1,95 = chọn đều)
    reward   phần thưởng của bước đó (chế độ đánh giá: không có thưởng cờ, không cắt lượt khi đứng yên)
và lý do lượt kết thúc (flag / enemy / pit / time / cap, đọc từ RAM như death_look.py). Các số đọc từ info và RAM
chỉ dùng để gắn nhãn kết quả; mạng vẫn chỉ nhận 4 khung hình 84x84.

Phần phân tích gom các lượt hỏng thành cụm theo x cuối (hai lượt cùng cụm nếu x cuối cách nhau không quá --gap), rồi
với mỗi cụm so V của 60 bước cuối ở các lượt hỏng tại đó với V của các lượt đi qua được, xét ở cùng tọa độ x. Các
lượt đứng yên tới hết giờ (time, cap) được xét riêng: V và phân phối hành động trong lúc kẹt.

Tệp ghi ra trong --out-dir:
    trace.npz              mọi bước của mọi lượt (mảng phẳng, cột `ep` cho biết bước thuộc lượt nào) và bảng từng lượt
    deaths.json            cùng định dạng với death_look.py, để đối chiếu x cuối
    summary.txt            các con số của từng cụm
    clusters.json          cùng các con số đó ở dạng máy đọc được
    overview.png           V dọc theo màn và nơi các lượt kết thúc
    cluster-x<lo>.png      mỗi cụm một hình
    stuck-x<lo>.png        mỗi cụm có lượt đứng yên tới hết giờ một hình, kèm các lượt thoát được khỏi chỗ đó
Ở cụm có lượt kẹt, summary.txt còn đếm số lượt tới nơi, số lượt thoát, mỗi lượt thoát đã đứng bao lâu và số lần
thoát trên 1000 bước đứng không tiến.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

ACTION_NAMES = ["đứng yên", "phải", "phải+nhảy", "phải+chạy", "phải+chạy+nhảy", "nhảy tại chỗ", "trái"]
ACTION_COLORS = ["#8c8c8c", "#9ecae1", "#4292c6", "#fdae6b", "#e6550d", "#74c476", "#9e5ac8"]
Y_PAGE = 0x00B5              # RAM: trang màn hình theo chiều dọc của Mario; > 1 nghĩa là đã rơi khỏi màn hình
X_BIN = 16                   # bề rộng một ô x (điểm ảnh) khi tính "V thường thấy ở chỗ này"
LAST = 60                    # số bước cuối của lượt hỏng được đem so
DROP = 0.5                   # "tụt rõ": thấp hơn mức thường thấy ở cùng x ít nhất từng này (V cỡ 3-4 khi chạy bình thường)
K_REPORT = (1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 60)
STUCK_STEPS = 200            # lượt được coi là "kẹt" nếu ở cuối lượt x xa nhất đã không tăng trong từng này bước
PAST = 16                    # "đã vượt" chỗ kẹt = x lớn hơn mép phải của cụm ít nhất từng này điểm ảnh (sát tường x có
                             # thể nhích quá mép phải vài điểm ảnh rồi lại bị đẩy về)
STUCK_WINDOWS = ((-10, 0), (1, 20), (21, 100), (101, 200), (201, 1000), (1001, 10 ** 6))


# ---------------------------------------------------------------------------------------------------------
# Phần 1: chạy lại các lượt và ghi từng bước
# ---------------------------------------------------------------------------------------------------------

def cause_of(env, info, done):
    """Chép từ death_look.cause_of."""
    if info.get("flag_get", False):
        return "flag"
    if not done:
        return "cap"
    if int(info.get("time", 1)) == 0:
        return "time"
    if int(env.unwrapped.ram[Y_PAGE]) > 1:
        return "pit"
    return "enemy"


def trace(ckpt, level, episodes=100, seed=7, max_steps=3000, max_noop=30):
    """Trả (dict các mảng theo bước, dict các mảng theo lượt, thông tin chung). Vòng lặp chép từ evaluate.run_episodes;
    model.act được viết bung ra để lấy thêm logits và V, thứ tự dùng số ngẫu nhiên của torch không đổi."""
    import torch
    from mario_ppo.env import make_env
    from mario_ppo.evaluate import NOOP, load_model

    model, ck = load_model(ckpt)
    rng = np.random.default_rng(seed)
    noops = [int(n) for n in rng.integers(0, max_noop + 1, size=episodes)]
    env = make_env(level, **dict(ck["config"]["env"], eval_mode=True))
    model.eval()
    torch.manual_seed(seed)
    cols = {k: [] for k in ("ep", "t", "v", "x", "y", "time", "action", "prob", "entropy", "reward", "probs")}
    eps = {k: [] for k in ("x_end", "steps", "noop_steps", "cause")}
    try:
        for i, n in enumerate(noops):
            obs, info = env.reset(seed=seed + i)
            done, steps = False, 0
            for _ in range(n):
                obs, _, term, trunc, info = env.step(NOOP)
                if term or trunc:
                    done = True
                    break
            while not done and steps < max_steps:
                with torch.no_grad():
                    logits, value = model(torch.as_tensor(np.asarray(obs)[None]))
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                a = int(action)
                probs = dist.probs[0].numpy()
                x_seen, y_seen, time_seen = info.get("x_pos", 40), info.get("y_pos", 79), info.get("time", 400)
                obs, reward, term, trunc, info = env.step(a)
                done, steps = term or trunc, steps + 1
                for k, val in (("ep", i), ("t", steps - 1), ("v", float(value)), ("x", int(x_seen)),
                               ("y", int(y_seen)), ("time", int(time_seen)), ("action", a), ("prob", float(probs[a])),
                               ("entropy", float(dist.entropy())), ("reward", float(reward)), ("probs", probs.copy())):
                    cols[k].append(val)
            for k, val in (("x_end", int(info.get("x_pos", 0))), ("steps", steps), ("noop_steps", n),
                           ("cause", cause_of(env, info, done))):
                eps[k].append(val)
            print(f"lượt {i:3d}: x cuối {eps['x_end'][-1]:5d}, {steps:4d} bước, {eps['cause'][-1]}", flush=True)
    finally:
        env.close()
    kinds = {"ep": np.int16, "t": np.int16, "x": np.int16, "y": np.int16, "time": np.int16, "action": np.int8}
    steps_arrays = {k: np.asarray(v, dtype=kinds.get(k, np.float32)) for k, v in cols.items()}
    ep_arrays = {"x_end": np.asarray(eps["x_end"]), "steps": np.asarray(eps["steps"]),
                 "noop_steps": np.asarray(eps["noop_steps"]), "cause": np.asarray(eps["cause"])}
    env_cfg, algo = ck["config"]["env"], ck["config"].get("algo", {})
    meta = {"level": level, "checkpoint": str(ckpt), "global_step": int(ck.get("global_step", 0)), "seed": seed,
            "episodes": episodes, "max_steps": max_steps, "gamma": float(algo.get("gamma", 0.9)),
            "flag_bonus": float(env_cfg.get("flag_bonus", 0.0)),
            "max_no_progress_steps": int(env_cfg.get("max_no_progress_steps", 0))}
    return steps_arrays, ep_arrays, meta


def save_trace(out, steps_arrays, ep_arrays, meta):
    np.savez_compressed(out / "trace.npz", meta=json.dumps(meta), **steps_arrays,
                        **{f"ep_{k}": v for k, v in ep_arrays.items()})
    rows = [{"episode": i, "x_pos": int(ep_arrays["x_end"][i]), "steps": int(ep_arrays["steps"][i]),
             "noop_steps": int(ep_arrays["noop_steps"][i]), "cause": str(ep_arrays["cause"][i])}
            for i in range(len(ep_arrays["x_end"]))]
    (out / "deaths.json").write_text(json.dumps(rows, indent=1))


class Trace:
    """trace.npz đọc lại, kèm các đại lượng tính thêm cho từng bước."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        for k in ("ep", "t", "v", "x", "y", "time", "action", "prob", "entropy", "reward", "probs"):
            setattr(self, k, z[k])
        self.x_end, self.n_steps = z["ep_x_end"], z["ep_steps"]
        self.noop_steps, self.cause = z["ep_noop_steps"], z["ep_cause"].astype(str)
        self.n_ep = len(self.x_end)
        self.start = np.concatenate([[0], np.cumsum(self.n_steps)])          # bước đầu của từng lượt trong mảng phẳng
        gamma, bonus = self.meta["gamma"], self.meta["flag_bonus"]
        self.k = np.zeros(len(self.v), dtype=np.int32)                       # số bước còn lại tới hết lượt (1 = bước cuối)
        self.ret = np.zeros(len(self.v), dtype=np.float32)                   # lợi tức chiết khấu thật tính lại từ phần thưởng
        self.since = np.zeros(len(self.v), dtype=np.int32)                   # số bước kể từ lần cuối x xa nhất tăng
        self.last_progress = np.zeros(self.n_ep, dtype=np.int32)             # bước cuối cùng mà x xa nhất còn tăng
        for e in range(self.n_ep):
            s = self.rows(e)
            n = s.stop - s.start
            self.k[s] = n - np.arange(n)
            r = self.reward[s].astype(np.float64).copy()
            if n and self.cause[e] == "flag":
                r[-1] += bonus        # lúc huấn luyện bước chạm cờ được cộng thưởng cờ; chế độ đánh giá không cộng
            g, acc = np.zeros(n), 0.0
            for j in range(n - 1, -1, -1):
                acc = r[j] + gamma * acc
                g[j] = acc
            self.ret[s] = g
            best, stalled = -10 ** 9, 0
            for j, xv in enumerate(self.x[s]):
                if xv > best:
                    best, stalled, self.last_progress[e] = xv, 0, j
                else:
                    stalled += 1
                self.since[s.start + j] = stalled
        self.stuck = np.array([self.cause[e] in ("time", "cap") or
                               (self.n_steps[e] > 0 and self.since[self.start[e + 1] - 1] >= STUCK_STEPS)
                               for e in range(self.n_ep)])

    def rows(self, e):
        return slice(int(self.start[e]), int(self.start[e + 1]))


# ---------------------------------------------------------------------------------------------------------
# Phần 2: gom cụm và so sánh
# ---------------------------------------------------------------------------------------------------------

def find_clusters(tr, gap, min_size):
    """Gom các lượt không chạm cờ theo x cuối: xếp tăng dần, tách cụm ở chỗ hai x cuối liên tiếp cách nhau > gap."""
    failed = sorted((int(tr.x_end[e]), e) for e in range(tr.n_ep) if tr.cause[e] != "flag")
    groups, current = [], []
    for xv, e in failed:
        if current and xv - current[-1][0] > gap:
            groups.append(current)
            current = []
        current.append((xv, e))
    if current:
        groups.append(current)
    clusters = [{"lo": g[0][0], "hi": g[-1][0], "episodes": [e for _, e in g]} for g in groups if len(g) >= min_size]
    small = [e for g in groups if len(g) < min_size for _, e in g]
    return clusters, small


def baseline_by_x(tr, episodes, x_lo, x_hi, min_count=5):
    """V và entropy "thường thấy" theo x: lấy mọi bước của các lượt `episodes` có x trong [x_lo, x_hi], chia ô X_BIN.
    Trả dict ô -> (p10, trung vị, p90 của V, trung vị entropy, số bước, p10 của lợi tức thật G, trung vị G)."""
    xs, vs, hs, gs = [], [], [], []
    for e in episodes:
        s = tr.rows(e)
        keep = (tr.x[s] >= x_lo) & (tr.x[s] <= x_hi)
        xs.append(tr.x[s][keep]); vs.append(tr.v[s][keep]); hs.append(tr.entropy[s][keep]); gs.append(tr.ret[s][keep])
    if not xs:
        return {}
    xs, vs, hs, gs = np.concatenate(xs), np.concatenate(vs), np.concatenate(hs), np.concatenate(gs)
    table = {}
    for b in np.unique(xs // X_BIN):
        m = xs // X_BIN == b
        if m.sum() >= min_count:
            p10, p50, p90 = np.percentile(vs[m], [10, 50, 90])
            table[int(b)] = (float(p10), float(p50), float(p90), float(np.median(hs[m])), int(m.sum()),
                             float(np.percentile(gs[m], 10)), float(np.median(gs[m])))
    return table


def lookup(table, x):
    """Ô của x; nếu ô đó thiếu số liệu thì lấy ô gần nhất trong phạm vi 3 ô, không có thì None."""
    b = int(x) // X_BIN
    for d in (0, -1, 1, -2, 2, -3, 3):
        if b + d in table:
            return table[b + d]
    return None


def analyse_cluster(tr, cluster, margin):
    """So 60 bước cuối của các lượt hỏng trong cụm (trừ các lượt kẹt tới hết giờ) với các lượt đi qua được."""
    lo, hi = cluster["lo"], cluster["hi"]
    members = cluster["episodes"]
    dying = [e for e in members if not tr.stuck[e]]
    stuck = [e for e in members if tr.stuck[e]]
    passing = [e for e in range(tr.n_ep) if tr.cause[e] == "flag" or tr.x_end[e] > hi + margin]
    x_from = min([int(tr.x[tr.rows(e)][-LAST:].min()) for e in dying] + [lo]) - X_BIN
    table = baseline_by_x(tr, passing, x_from, hi + X_BIN)
    per_k = {}
    for k in range(1, LAST + 1):
        v, base, low, ret, ret_base, ret_low, ent, ent_base, prob = [], [], [], [], [], [], [], [], []
        for e in dying:
            if tr.n_steps[e] < k:
                continue
            j = int(tr.start[e + 1]) - k
            ref = lookup(table, tr.x[j])
            if ref is None:
                continue
            v.append(tr.v[j]); base.append(ref[1]); low.append(tr.v[j] < ref[0]); ret.append(tr.ret[j])
            ret_base.append(ref[6]); ret_low.append(tr.ret[j] < ref[5])
            ent.append(tr.entropy[j]); ent_base.append(ref[3]); prob.append(tr.prob[j])
        if v:
            per_k[k] = {"n": len(v), "v": float(np.median(v)), "v_q25": float(np.percentile(v, 25)),
                        "v_q75": float(np.percentile(v, 75)), "usual": float(np.median(base)),
                        "diff": float(np.median(np.array(v) - np.array(base))), "below_p10": float(np.mean(low)),
                        "ret": float(np.median(ret)), "ret_usual": float(np.median(ret_base)),
                        "ret_diff": float(np.median(np.array(ret) - np.array(ret_base))),
                        "ret_below_p10": float(np.mean(ret_low)), "entropy": float(np.median(ent)),
                        "entropy_usual": float(np.median(ent_base)), "prob": float(np.median(prob))}

    # Mạng "thấy trước" bao nhiêu bước: số bước k lớn nhất sao cho điều kiện đúng ở MỌI bước từ k về 1.
    #   steps_ahead      ít nhất một nửa số lượt hỏng có V thấp hơn p10 của các lượt đi qua ở cùng x
    #   drop_ahead_v     trung vị của (V - V thường thấy ở cùng x) thấp hơn -DROP
    #   drop_ahead_g     như trên nhưng với lợi tức thật G: mức "thấy trước" của một critic biết trước kết cục
    def lead(ok):
        ahead = 0
        for k in range(1, LAST + 1):
            if k in per_k and ok(per_k[k]):
                ahead = k
            else:
                break
        return ahead

    return {"lo": lo, "hi": hi, "n": len(members), "causes": dict(Counter(tr.cause[e] for e in members)),
            "dying": dying, "stuck": stuck, "n_passing": len(passing), "passing": passing, "table": table,
            "per_k": per_k, "steps_ahead": lead(lambda r: r["below_p10"] >= 0.5),
            "drop_ahead_v": lead(lambda r: r["diff"] <= -DROP), "drop_ahead_g": lead(lambda r: r["ret_diff"] <= -DROP)}


def analyse_stuck(tr, episodes):
    """V và phân phối hành động của các lượt kẹt, theo số bước kể từ lần cuối x xa nhất còn tăng (bước 0)."""
    windows = []
    for a, b in STUCK_WINDOWS:
        v, probs, back, n_ep = [], [], [], 0
        for e in episodes:
            s = tr.rows(e)
            rel = np.arange(s.stop - s.start) - tr.last_progress[e]
            m = (rel >= a) & (rel <= b)
            if m.any():
                n_ep += 1
                v.append(tr.v[s][m]); probs.append(tr.probs[s][m])
                back.append(float(np.max(tr.x[s].max() - tr.x[s][m])))
        if v:
            windows.append({"from": a, "to": b, "episodes": n_ep, "v_mean": float(np.concatenate(v).mean()),
                            "v_min": float(np.concatenate(v).min()), "v_max": float(np.concatenate(v).max()),
                            "probs": [float(p) for p in np.concatenate(probs).mean(axis=0)],
                            "max_back": float(np.max(back))})
    lengths = [int(tr.n_steps[e] - tr.last_progress[e]) for e in episodes]
    peaks = [int(tr.y[tr.rows(e)][tr.last_progress[e]:].max()) for e in episodes]
    return {"episodes": list(map(int, episodes)), "stuck_steps": lengths, "windows": windows,
            "y_peak": {"min": min(peaks), "median": float(np.median(peaks)), "max": max(peaks)}}


def escapes(tr, lo, hi, gap):
    """Các lượt tới được chỗ kẹt (x >= lo - gap) rồi vượt qua mép phải hi của cụm: mỗi lượt đã đứng không tiến bao
    nhiêu bước liên tục ngay trước khi vượt (cùng cách đếm với NoProgressTruncation lúc huấn luyện), V và độ cao y
    lúc vượt. Trả (danh sách lượt vượt, số lượt tới nơi, tổng số bước "không tiến" của mọi lượt ở chỗ này)."""
    out, reached, stalled_total = [], 0, 0
    for e in range(tr.n_ep):
        s = tr.rows(e)
        x = tr.x[s]
        if x.max() < lo - gap:
            continue
        reached += 1
        beyond = np.where(x > hi + PAST)[0]
        arrive = int(np.argmax(x >= lo - gap))
        if not len(beyond) or beyond[0] <= arrive:
            stalled_total += int((tr.since[s][arrive:] > 0).sum())
            continue
        stalled_total += int((tr.since[s][arrive:int(beyond[0])] > 0).sum())
        f = int(beyond[0])
        since, v = tr.since[s][arrive:f], tr.v[s][arrive:f]
        waiting = v[since > 10]
        at_wall = arrive + int(np.argmax(x[arrive:f] >= lo))             # lần đầu chạm chỗ kẹt
        leave = arrive + int(np.where(x[arrive:f] <= hi)[0][-1]) + 1      # bước rời chỗ kẹt hẳn
        xs = x[at_wall:leave].astype(int)
        out.append({"episode": e, "arrive": arrive, "beyond": f, "leave": leave, "wait": int(since.max()),
                    "back": int(np.max(np.maximum.accumulate(xs) - xs)) if len(xs) else 0,
                    "y_peak": int(tr.y[s][arrive:f + 3].max()),
                    "v_waiting": float(np.median(waiting)) if len(waiting) else None,
                    "v_last8": float(tr.v[s][max(arrive, leave - 8):leave + 1].max()), "cause": str(tr.cause[e])})
    return out, reached, stalled_total


# ---------------------------------------------------------------------------------------------------------
# Phần 3: hình và bảng chữ
# ---------------------------------------------------------------------------------------------------------

def dotted(n):
    """9998336 -> '9.998.336' (cách viết số của nhóm)."""
    return f"{int(n):,}".replace(",", ".")


def draw_overview(tr, clusters, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = baseline_by_x(tr, range(tr.n_ep), 0, 10 ** 5, min_count=3)
    bins = sorted(table)
    xs = [(b + 0.5) * X_BIN for b in bins]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    a1.fill_between(xs, [table[b][0] for b in bins], [table[b][2] for b in bins], color="#9ecae1", alpha=0.6,
                    label="80% số bước nằm trong dải này")
    a1.plot(xs, [table[b][1] for b in bins], color="#08519c", lw=1.2, label="trung vị V (mọi lượt)")
    a1.axhline(0, color="k", lw=0.5)
    a1.set_ylabel("V do critic dự đoán")
    a1.set_title(f"Màn {tr.meta['level']}, checkpoint bước {dotted(tr.meta['global_step'])}: V dọc theo màn "
                 f"({tr.n_ep} lượt, seed {tr.meta['seed']})")
    a1.legend(loc="lower left", fontsize=8)
    colors = {"enemy": "#cb181d", "pit": "#6a51a3", "time": "#fd8d3c", "cap": "#fdd0a2", "flag": "#31a354"}
    edges = np.arange(0, int(tr.x_end.max()) + 100, 50)
    bottom = np.zeros(len(edges) - 1)
    for cause in ("enemy", "pit", "time", "cap", "flag"):
        h, _ = np.histogram(tr.x_end[tr.cause == cause], bins=edges)
        if h.sum():
            a2.bar(edges[:-1], h, width=50, bottom=bottom, align="edge", color=colors[cause],
                   label=f"{cause} ({int(h.sum())})")
            bottom += h
    for c in clusters:
        for ax in (a1, a2):
            ax.axvspan(c["lo"] - 8, c["hi"] + 8, color="k", alpha=0.07)
    a2.set_ylabel("số lượt kết thúc")
    a2.set_xlabel("x của Mario (điểm ảnh); vùng xám = các cụm được phân tích")
    a2.legend(fontsize=8, ncol=5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_cluster(tr, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table, dying, per_k = res["table"], res["dying"], res["per_k"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    bins = sorted(table)
    xs = [(b + 0.5) * X_BIN for b in bins]
    ax.fill_between(xs, [table[b][0] for b in bins], [table[b][2] for b in bins], color="#bdbdbd", alpha=0.7,
                    label=f"{res['n_passing']} lượt đi qua: 80% số bước")
    ax.plot(xs, [table[b][1] for b in bins], color="#525252", lw=1.3, label="lượt đi qua: trung vị")
    for i, e in enumerate(dying):
        s = tr.rows(e)
        ax.plot(tr.x[s][-LAST:], tr.v[s][-LAST:], color="#cb181d", lw=0.7, alpha=0.55,
                label=f"{len(dying)} lượt hỏng ở đây ({LAST} bước cuối)" if i == 0 else None)
        ax.plot(tr.x[s][-1], tr.v[s][-1], "x", color="#67000d", ms=5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("x của Mario (điểm ảnh); dấu x = bước cuối của lượt")
    ax.set_ylabel("V do critic dự đoán")
    ax.set_title("V theo vị trí")
    ax.legend(fontsize=8, loc="lower left")

    ks = sorted(per_k)
    neg = [-k for k in ks]
    ax = axes[1]
    ax.fill_between(neg, [per_k[k]["v_q25"] for k in ks], [per_k[k]["v_q75"] for k in ks], color="#fcbba1", alpha=0.7)
    ax.plot(neg, [per_k[k]["v"] for k in ks], color="#cb181d", lw=1.8, label="lượt hỏng: V (trung vị, dải 25-75%)")
    ax.plot(neg, [per_k[k]["usual"] for k in ks], color="#525252", lw=1.5,
            label="V thường thấy ở cùng x (lượt đi qua)")
    ax.plot(neg, [per_k[k]["ret"] for k in ks], color="#08519c", lw=1.2, ls="--",
            label="lợi tức thật G của lượt hỏng (tính lại từ phần thưởng)")
    ax.plot(neg, [per_k[k]["ret_usual"] for k in ks], color="#969696", lw=1.0, ls="--",
            label="G thường thấy ở cùng x (lượt đi qua)")
    ax.axhline(0, color="k", lw=0.5)
    if res["drop_ahead_v"]:
        ax.axvline(-res["drop_ahead_v"], color="#cb181d", lw=0.8, ls=":")
    if res["drop_ahead_g"]:
        ax.axvline(-res["drop_ahead_g"], color="#08519c", lw=0.8, ls=":")
    ax.set_xlabel("số bước tác tử trước khi lượt kết thúc (1 bước = 4 khung hình)")
    ax.set_title(f"V tụt hơn {DROP} so với thường lệ từ {res['drop_ahead_v']} bước trước khi chết "
                 f"(G thật: từ {res['drop_ahead_g']})", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[2]
    ax.plot(neg, [per_k[k]["entropy"] for k in ks], color="#cb181d", lw=1.6, label="lượt hỏng: entropy (trung vị)")
    ax.plot(neg, [per_k[k]["entropy_usual"] for k in ks], color="#525252", lw=1.4,
            label="entropy thường thấy ở cùng x")
    ax.plot(neg, [per_k[k]["prob"] for k in ks], color="#31a354", lw=1.2, ls="--",
            label="xác suất của hành động đã chọn (lượt hỏng)")
    ax.set_ylim(0, 2.0)
    ax.set_xlabel("số bước tác tử trước khi lượt kết thúc")
    ax.set_title("Chính sách có phân vân không (entropy tối đa 1,95)")
    ax.legend(fontsize=8, loc="upper left")
    causes = ", ".join(f"{n} {c}" for c, n in sorted(res["causes"].items(), key=lambda kv: -kv[1]))
    fig.suptitle(f"Màn {tr.meta['level']}, cụm x {res['lo']}-{res['hi']}: {res['n']} lượt kết thúc ({causes}); "
                 f"checkpoint bước {dotted(tr.meta['global_step'])}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def draw_stuck(tr, res, episodes, path, esc=(), span=600):
    """Bốn ô: V, xác suất hành động, x so với mép phải của cụm, độ cao y. Lượt kẹt màu cam/tím, tính từ lần tiến cuối;
    lượt thoát được (esc) màu xanh lá, tính từ lúc tới nơi (x >= mép trái của cụm - gap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    grid = np.arange(-40, span + 1)
    v_all = np.full((len(episodes), len(grid)), np.nan)
    p_all = np.full((len(episodes), len(grid), tr.probs.shape[1]), np.nan)
    for i, e in enumerate(episodes):
        s = tr.rows(e)
        rel = np.arange(s.stop - s.start) - tr.last_progress[e]
        m = (rel >= grid[0]) & (rel <= grid[-1])
        v_all[i, rel[m] - grid[0]] = tr.v[s][m]
        p_all[i, rel[m] - grid[0]] = tr.probs[s][m]
        axes[0].plot(rel[m], tr.v[s][m], color="#fd8d3c", lw=0.5, alpha=0.4)
        axes[2].plot(rel[m], tr.x[s][m].astype(int) - res["hi"], color="#6a51a3", lw=0.5, alpha=0.4)
        axes[3].plot(rel[m], tr.y[s][m], color="#6a51a3", lw=0.5, alpha=0.25)
    axes[0].plot(grid, np.nanmedian(v_all, axis=0), color="#7f2704", lw=1.6, label="trung vị các lượt kẹt")
    for i, r in enumerate(esc):
        s = tr.rows(r["episode"])
        rel = np.arange(s.stop - s.start) - r["arrive"]
        m = (rel >= grid[0]) & (rel <= min(grid[-1], r["beyond"] - r["arrive"] + 30))
        label = "lượt thoát được" if i == 0 else None
        axes[0].plot(rel[m], tr.v[s][m], color="#238b45", lw=1.4, label=label)
        axes[2].plot(rel[m], tr.x[s][m].astype(int) - res["hi"], color="#238b45", lw=1.4, label=label)
        axes[3].plot(rel[m], tr.y[s][m], color="#238b45", lw=1.2, label=label)
    limit = tr.meta["max_no_progress_steps"]
    for ax in axes:
        ax.axvline(0, color="k", lw=0.6)
        if limit:
            ax.axvline(limit, color="#cb181d", lw=0.9, ls="--")
        ax.set_xlabel("số bước kể từ lần cuối Mario còn tiến (bước 0)")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("V do critic dự đoán")
    axes[0].set_title(f"V trong lúc kẹt ({len(episodes)} lượt kẹt)")
    axes[0].legend(fontsize=8)
    mean_p = np.nanmean(p_all, axis=0)
    kernel = np.ones(15) / 15
    weight = np.convolve(np.ones(len(grid)), kernel, mode="same")      # ở hai mép cửa sổ làm trơn bị hụt, chia lại cho đủ
    smooth = np.stack([np.convolve(np.nan_to_num(mean_p[:, a]), kernel, mode="same") / weight
                       for a in range(mean_p.shape[1])])
    axes[1].stackplot(grid, smooth, labels=ACTION_NAMES, colors=ACTION_COLORS)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Xác suất hành động, lượt kẹt (trung bình, làm trơn 15 bước)", fontsize=10)
    axes[1].legend(fontsize=7, loc="lower right", ncol=2)
    axes[2].set_ylim(-60, 60)
    axes[2].set_ylabel(f"x trừ {res['hi']} (điểm ảnh)")
    axes[2].set_title("Mario có lùi lại không")
    axes[2].legend(fontsize=8)
    axes[3].set_ylabel("độ cao y của Mario (đọc từ info; lớn = cao)")
    axes[3].set_title("Mario nhảy cao tới đâu")
    axes[3].legend(fontsize=8)
    note = f"; đường đỏ đứt = bước {limit}, nơi lượt bị cắt lúc huấn luyện" if limit else ""
    fig.suptitle(f"Màn {tr.meta['level']}, cụm x {res['lo']}-{res['hi']}: các lượt đứng tới hết giờ và "
                 f"{len(esc)} lượt thoát được{note}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def cluster_lines(tr, res, stuck_res, esc=None, reached=0, stalled_total=0):
    causes = ", ".join(f"{c} {n}" for c, n in sorted(res["causes"].items(), key=lambda kv: -kv[1]))
    lines = ["", f"--- cụm x {res['lo']}-{res['hi']}: {res['n']} lượt ({causes}); {res['n_passing']} lượt đi qua được ---"]
    if res["per_k"]:
        lines.append(f"{len(res['dying'])} lượt hỏng không kẹt (lượt số {', '.join(map(str, res['dying']))})")
        lines.append("  k = số bước trước khi lượt kết thúc; V = trung vị V của các lượt hỏng; thường = trung vị V của các")
        lines.append("  lượt đi qua ở cùng x; dưới p10 = phần lượt hỏng có V thấp hơn 90% các lần đi qua; G = lợi tức thật")
        lines.append("  (G thường = trung vị G của các lượt đi qua ở cùng x; G-thường = trung vị của hiệu từng lượt)")
        lines.append("     k    n      V   thường   V-thường  dưới p10      G  G thường  G-thường   entropy  (thường)  p(hành động)")
        for k in K_REPORT:
            if k in res["per_k"]:
                r = res["per_k"][k]
                lines.append(f"  {k:4d}  {r['n']:3d}  {r['v']:6.2f}  {r['usual']:6.2f}   {r['diff']:7.2f}   "
                             f"{r['below_p10']:6.0%}  {r['ret']:6.2f}   {r['ret_usual']:6.2f}   {r['ret_diff']:7.2f}"
                             f"   {r['entropy']:6.2f}   {r['entropy_usual']:6.2f}     {r['prob']:6.2f}")
        lines.append(f"  V thấp hơn thường lệ (ít nhất nửa số lượt dưới p10) liên tục từ {res['steps_ahead']} bước trước khi kết thúc")
        lines.append(f"  V thấp hơn thường lệ ít nhất {DROP} (trung vị) liên tục từ {res['drop_ahead_v']} bước trước; "
                     f"G thật theo cùng tiêu chí: từ {res['drop_ahead_g']} bước trước")
    if stuck_res:
        lines.append(f"{len(stuck_res['episodes'])} lượt kẹt (lượt số {', '.join(map(str, stuck_res['episodes']))}); "
                     f"số bước đứng lại sau lần tiến cuối: {min(stuck_res['stuck_steps'])}-{max(stuck_res['stuck_steps'])}")
        lines.append("  khoảng bước (0 = lần tiến cuối)   lượt  V tb   V min  V max  lùi xa nhất   xác suất trung bình: "
                     + " | ".join(ACTION_NAMES))
        for w in stuck_res["windows"]:
            to = "hết" if w["to"] >= 10 ** 6 else f"{w['to']}"
            lines.append(f"  {w['from']:6d} .. {to:>5s}                 {w['episodes']:3d}  {w['v_mean']:5.2f}  {w['v_min']:5.2f}  "
                         f"{w['v_max']:5.2f}   {w['max_back']:6.0f} px    " + " | ".join(f"{p:.2f}" for p in w["probs"]))
        yp = stuck_res["y_peak"]
        lines.append(f"  độ cao y lớn nhất của từng lượt kẹt sau lần tiến cuối: nhỏ nhất {yp['min']}, trung vị {yp['median']:.0f}, "
                     f"lớn nhất {yp['max']}")
    if esc is not None:
        limit = tr.meta["max_no_progress_steps"]
        lines.append(f"  {reached} lượt tới được chỗ này (x >= {res['lo']} - gap); {len(esc)} lượt vượt qua x {res['hi']} + {PAST}; "
                     f"trong đó {sum(r['wait'] <= limit for r in esc)} lượt vượt sau <= {limit} bước không tiến, "
                     f"{sum(r['wait'] > limit for r in esc)} lượt sau > {limit} bước")
        rate = f"{1000 * len(esc) / stalled_total:.3f}" if stalled_total else "–"
        lines.append(f"  tổng số bước không tiến ở chỗ này (mọi lượt tới nơi): {stalled_total}; "
                     f"số lần thoát trên 1000 bước không tiến: {rate}")
        slow = [r for r in esc if r["wait"] >= 10]
        lines.append(f"  {len(esc) - len(slow)} lượt vượt mà không đứng quá 9 bước; các lượt vượt sau khi đứng từ 10 bước "
                     f"trở lên ({len(slow)}):")
        for r in slow:
            vw = "–" if r["v_waiting"] is None else f"{r['v_waiting']:.2f}"
            lines.append(f"    lượt {r['episode']:3d} ({r['cause']}): tới ở bước {r['arrive']}, rời chỗ kẹt ở bước {r['leave']}, "
                         f"đứng không tiến tối đa {r['wait']} bước liên tục, lùi xa nhất {r['back']} px, y cao nhất {r['y_peak']}, "
                         f"V trung vị lúc đứng {vw}, V lớn nhất trong 8 bước trước khi rời {r['v_last8']:.2f}")
    return lines


def analyse(out, gap, min_size, margin):
    """Gom cụm, so sánh, vẽ hình, ghi summary.txt và clusters.json từ trace.npz trong `out`."""
    tr = Trace(out / "trace.npz")
    m = tr.meta
    clusters, small = find_clusters(tr, gap, min_size)
    causes = Counter(tr.cause)
    lines = [f"=== {m['level']} với {m['checkpoint']} (bước {m['global_step']}): {tr.n_ep} lượt, seed {m['seed']} ===",
             "lý do: " + ", ".join(f"{c} {n}" for c, n in causes.most_common()),
             f"x cuối trung bình {tr.x_end.mean():.0f}; tổng {len(tr.v)} bước tác tử",
             f"V trên mọi bước: trung bình {tr.v.mean():.2f}, p10 {np.percentile(tr.v, 10):.2f}, "
             f"p90 {np.percentile(tr.v, 90):.2f}; phần thưởng trung bình mỗi bước {tr.reward.mean():.3f}",
             f"sai số của critic |V - G| trên mọi bước: trung vị {np.median(np.abs(tr.v - tr.ret)):.2f} "
             f"(G = lợi tức chiết khấu thật, gamma {m['gamma']}, có cộng thưởng cờ {m['flag_bonus']})",
             f"gom cụm: hai x cuối cùng cụm nếu cách nhau <= {gap}; cụm phải có >= {min_size} lượt; "
             f"lượt 'đi qua' = chạm cờ hoặc x cuối > mép phải của cụm + {margin}",
             f"{len(clusters)} cụm chứa {sum(len(c['episodes']) for c in clusters)} lượt; {len(small)} lượt hỏng lẻ "
             f"không thuộc cụm nào (x cuối: {', '.join(str(int(tr.x_end[e])) for e in sorted(small, key=lambda e: tr.x_end[e]))})"]
    report = []
    draw_overview(tr, clusters, out / "overview.png")
    for c in clusters:
        res = analyse_cluster(tr, c, margin)
        stuck_res = analyse_stuck(tr, res["stuck"]) if res["stuck"] else None
        esc, reached, stalled_total = escapes(tr, res["lo"], res["hi"], gap) if res["stuck"] else (None, 0, 0)
        if res["per_k"]:
            draw_cluster(tr, res, out / f"cluster-x{res['lo']}.png")
        if stuck_res:
            draw_stuck(tr, res, res["stuck"], out / f"stuck-x{res['lo']}.png", esc)
        lines += cluster_lines(tr, res, stuck_res, esc, reached, stalled_total)
        report.append({"lo": res["lo"], "hi": res["hi"], "n": res["n"], "causes": res["causes"],
                       "dying": list(map(int, res["dying"])), "stuck": list(map(int, res["stuck"])),
                       "n_passing": res["n_passing"], "steps_ahead": res["steps_ahead"],
                       "drop_ahead_v": res["drop_ahead_v"], "drop_ahead_g": res["drop_ahead_g"],
                       "per_k": {str(k): v for k, v in res["per_k"].items()}, "stuck_analysis": stuck_res,
                       "reached": reached, "escapes": esc, "stalled_steps_total": stalled_total})
    (out / "clusters.json").write_text(json.dumps({"meta": m, "gap": gap, "min_size": min_size, "margin": margin,
                                                   "clusters": report}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--level", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--reuse", action="store_true", help="không chạy lại trò chơi, chỉ phân tích trace.npz đã có")
    p.add_argument("--gap", type=int, default=48, help="hai x cuối cùng cụm nếu cách nhau không quá từng này điểm ảnh")
    p.add_argument("--min-size", type=int, default=5, help="số lượt tối thiểu để một cụm được phân tích")
    p.add_argument("--margin", type=int, default=100, help="lượt 'đi qua' phải kết thúc xa hơn mép phải của cụm từng này")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.reuse:
        if not args.ckpt:
            p.error("cần --ckpt (hoặc dùng --reuse với trace.npz đã có)")
        save_trace(out, *trace(args.ckpt, args.level, args.episodes, args.seed, args.max_steps))
    analyse(out, args.gap, args.min_size, args.margin)


if __name__ == "__main__":
    main()
