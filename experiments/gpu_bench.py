"""Đo tốc độ (không huấn luyện mạng nào được giữ lại): bước cập nhật PPO và lượt suy luận lúc thu dữ liệu, trên một thiết bị.

    python experiments/gpu_bench.py cpu|cuda [--threads 6] [--reps 3]

In một dòng JSON. Batch là 4.096 quan sát uint8 ngẫu nhiên (8 x 512 như cấu hình của nhóm); lần cập nhật chạy đủ
10 epoch x 16 mini-batch (tắt target_kl), nên hai thiết bị cùng làm 160 bước gradient. Không có bước môi trường nào.
Bản torch có CUDA nằm ở venv riêng ~/.venvs/mario-ppo-gpu; số đo ngày 27/11/2026 ở docs/config-review.md, Mục 5.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mario_ppo.model import ActorCritic  # noqa: E402
from mario_ppo.ppo import ppo_update  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("device")
p.add_argument("--threads", type=int, default=6)
p.add_argument("--reps", type=int, default=3)
args = p.parse_args()

torch.set_num_threads(args.threads)
torch.manual_seed(0)
dev = torch.device(args.device)
cuda = dev.type == "cuda"


def sync():
    if cuda:
        torch.cuda.synchronize()


cfg = SimpleNamespace(update_epochs=10, num_minibatches=16, clip_coef=0.15, clip_vloss=False, norm_adv=True,
                      ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, target_kl=None, lr=2e-4, anneal_lr=True)
model = ActorCritic(7, in_channels=4).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-6, eps=1e-5)  # lr rất nhỏ: chỉ đo thời gian
n = 4096
cpu_batch = {"obs": torch.randint(0, 256, (n, 4, 84, 84), dtype=torch.uint8), "actions": torch.randint(0, 7, (n,)),
             "old_logp": torch.full((n,), -1.9459), "old_values": torch.randn(n), "advantages": torch.randn(n),
             "returns": torch.randn(n)}

# làm nóng (cuDNN, bộ cấp phát), không tính giờ
warm = {k: v[:512].to(dev) for k, v in cpu_batch.items()}
ppo_update(model, opt, warm, SimpleNamespace(**{**vars(cfg), "update_epochs": 1, "num_minibatches": 2}))
sync()

transfer, update = [], []
for _ in range(args.reps):
    t0 = time.perf_counter()
    batch = {k: v.to(dev) for k, v in cpu_batch.items()}
    sync()
    t1 = time.perf_counter()
    model.train()
    stats = ppo_update(model, opt, batch, cfg)
    sync()
    t2 = time.perf_counter()
    transfer.append(t1 - t0)
    update.append(t2 - t1)
    assert stats["minibatch_updates"] == 160

# suy luận lúc thu dữ liệu: model.act trên 8 quan sát, train.py gọi một lần mỗi bước tác tử
model.eval()
obs8 = torch.randint(0, 256, (8, 4, 84, 84), dtype=torch.uint8)
for _ in range(50):
    model.act(obs8)
sync()
t0 = time.perf_counter()
calls = 2048
for _ in range(calls):
    model.act(obs8)
sync()
act = (time.perf_counter() - t0) / calls

med = lambda xs: sorted(xs)[len(xs) // 2]
out = {"device": args.device, "torch": torch.__version__, "threads": args.threads,
       "update_s": round(med(update), 3), "update_all": [round(u, 3) for u in update],
       "transfer_s": round(med(transfer), 4), "act_ms_per_call": round(act * 1000, 3),
       "act_s_per_512_calls": round(act * 512, 3)}
if cuda:
    out["gpu"] = torch.cuda.get_device_name(0)
    out["gpu_mem_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
print(json.dumps(out))
