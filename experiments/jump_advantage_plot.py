"""Figure for docs/optimize-1-1.md, Mục 18: advantage of the correct 1-3 jump over the trained policy's jump, by gamma,
under three reward rules. Numbers come from experiments/jump_returns.py at the 819k checkpoint of
runs/1-3/tuning/first-819k (logs: eval/returns-s0-819k-v2.txt for P, S+P; eval/returns-s0-819k-sb.json for S+B).

    python experiments/jump_advantage_plot.py runs/1-3/tuning/first-819k/eval/returns-s0-819k-sb.json docs/reports/figures/1-3-jump-advantage.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAMMAS = [0.9, 0.95, 0.97, 0.98, 0.99]
# returns-s0-819k-v2.txt: P (policy, 50 rollouts, all identical) and S+P (scripted jump then policy, 200 rollouts)
P = {"R0": [2.83, 3.55, 3.88, 4.06, 4.24], "Rclaw": [1.80, 1.23, 0.72, 0.37, -0.06],
     "Rland": [0.70, 0.30, 0.03, -0.14, -0.34]}
SP = {"R0": [1.30, 1.48, 1.58, 1.65, 1.75], "Rclaw": [1.27, 1.40, 1.45, 1.48, 1.51],
      "Rland": [0.57, 0.87, 1.07, 1.20, 1.36]}
STYLE = {"R0": ("#52514e", "thưởng hiện tại"), "Rclaw": ("#eb6834", "không cộng x khi rơi hố"),
         "Rland": ("#3a7bd5", "cộng x khi chạm đất")}

sb = json.load(open(sys.argv[1]))["SB"]["G"]
SB = {rule: [sb[f"{rule}|{g}"] for g in GAMMAS] for rule in P}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
panels = [(axes[0], SP, "Sau khi đáp, đi tiếp bằng chính sách hiện tại\n(186/200 lượt chết vì nấm ở x ≈ 684)"),
          (axes[1], SB, "Sau khi đáp, đi tiếp giỏi\n(đường tốt nhất trong 2.000 lượt ngẫu nhiên, tới x 1505)")]
for ax, branch, title in panels:
    for rule, (color, label) in STYLE.items():
        adv = [b - p for b, p in zip(branch[rule], P[rule])]
        ax.plot(GAMMAS, adv, color=color, linewidth=2.2, marker="o", markersize=5, markeredgecolor="white",
                label=label)
        dy = {"R0": 0, "Rclaw": -6, "Rland": 6}[rule]
        ax.annotate(f"{adv[-1]:+.1f}", (GAMMAS[-1], adv[-1]), xytext=(6, dy), textcoords="offset points",
                    va="center", fontsize=8.5, color="#2b2a27")
    ax.axhline(0, color="#898781", linewidth=1)
    ax.axvline(0.9, color="#c3c2b7", linestyle=(0, (1, 3)), linewidth=1)
    ax.text(0.9, ax.get_ylim()[1], " γ đang dùng", va="top", fontsize=8, color="#898781")
    ax.set_title(title, loc="left", fontsize=9.5, color="#2b2a27")
    ax.set_xlabel("γ", color="#52514e")
    ax.set_xticks(GAMMAS)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", length=0)
axes[0].set_ylabel("Lợi thế của cú nhảy đúng\nE[G | nhảy đúng] − E[G | chính sách]", color="#52514e")
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 0.93))
fig.suptitle("1-3, trạng thái trên bệ thấp (x 597), checkpoint 819k: trên 0 nghĩa là PPO sẽ học cú nhảy đúng",
             x=0.01, ha="left", fontsize=10.5, color="#2b2a27")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=130)
print("saved", sys.argv[2])
