"""Kiểm tra của nhóm cho mario_ppo/train_joint.py (huấn luyện một mạng trên nhiều màn).

    python -m pytest -q tests_group/test_train_joint.py

Dùng trình giả lập thật, ba lượt huấn luyện tí hon (khoảng một phút).
"""
import csv

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mario_ppo.train_joint import assign_levels

PER_LEVEL = ("episodes", "ep_return", "ep_length", "x_pos", "flag_rate")


def _tiny_cfg(tmp_path, name, **extra):
    """Cấu hình 4-1 của nhóm (có mẹo huấn luyện và lịch entropy) thu nhỏ còn 256 bước."""
    cfg = OmegaConf.load("configs/group-4-1.yaml")
    over = OmegaConf.from_dotlist([
        f"run_dir={tmp_path}/{name}", "level=1-1", "torch_threads=2", "algo.num_envs=2", "algo.num_steps=32",
        "algo.num_minibatches=2", "algo.update_epochs=2", "algo.total_steps=256", "checkpoint_every=128",
        "eval_episodes=1", "eval_max_steps=20", "video=false",
        "env.max_no_progress_steps=5",  # cắt lượt sớm để có lượt kết thúc và đi qua nhánh xử lý lượt bị cắt
    ])
    return OmegaConf.merge(cfg, over, extra)


def _rows(run, name="progress.csv"):
    with open(run / name, newline="") as handle:
        return list(csv.DictReader(handle))


def test_assign_levels_alternates_and_validates():
    assert assign_levels(["1-1", "4-1"], 8) == ["1-1", "4-1"] * 4
    assert assign_levels(["1-1"], 3) == ["1-1"] * 3
    with pytest.raises(ValueError):
        assign_levels(["1-1", "4-1"], 3)   # không chia đều được
    with pytest.raises(ValueError):
        assign_levels(["1-1", "1-1"], 2)   # trùng màn
    with pytest.raises(ValueError):
        assign_levels(["9-1"], 2)          # màn không tồn tại


def test_single_level_gives_the_same_numbers_as_train(tmp_path):
    """Vòng lặp chép từ train.py: với một màn, mọi con số và trọng số phải trùng train.py của thầy."""
    from mario_ppo.train import train
    from mario_ppo.train_joint import train as train_joint
    base = train(_tiny_cfg(tmp_path, "base"))
    joint = train_joint(_tiny_cfg(tmp_path, "joint", levels=["1-1"]))

    rows_base, rows_joint = _rows(base), _rows(joint)
    assert len(rows_base) == len(rows_joint) == 4
    assert int(rows_base[-1]["episodes"]) > 0, "cần có lượt kết thúc để phép so sánh có nghĩa"
    for key in rows_base[0]:
        if key == "sps":  # tốc độ phụ thuộc đồng hồ
            continue
        a = np.array([float(r[key]) for r in rows_base])
        b = np.array([float(r[key]) for r in rows_joint])
        np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-9, equal_nan=True, err_msg=key)
    for key in PER_LEVEL:  # cột riêng của màn duy nhất trùng cột chung
        assert [r[f"{key}_1-1"] for r in rows_joint] == [r[key] for r in rows_joint]

    assert [r["flag_rate"] for r in _rows(base, "evals.csv")] == [r["flag_rate"] for r in _rows(joint, "evals.csv")]
    assert [r["x_pos"] for r in _rows(base, "evals.csv")] == [r["x_pos"] for r in _rows(joint, "evals.csv")]
    ck_base = torch.load(sorted((base / "checkpoints").glob("*.pt"))[-1], weights_only=False)
    ck_joint = torch.load(sorted((joint / "checkpoints").glob("*.pt"))[-1], weights_only=False)
    assert ck_base["global_step"] == ck_joint["global_step"] == 256
    for name, weight in ck_base["model"].items():
        assert torch.allclose(ck_joint["model"][name], weight, rtol=1e-6, atol=1e-8), name


def test_two_levels_log_and_evaluate_each_level(tmp_path):
    from mario_ppo.evaluate import evaluate_checkpoint
    from mario_ppo.train_joint import train as train_joint
    run = train_joint(_tiny_cfg(tmp_path, "two", levels=["1-1", "4-1"]))

    rows = _rows(run)
    assert len(rows) == 4
    for level in ("1-1", "4-1"):
        for key in PER_LEVEL:
            assert f"{key}_{level}" in rows[0]
        assert int(rows[-1][f"episodes_{level}"]) > 0  # môi trường của màn này đã chạy và được ghi đúng màn
    last = rows[-1]
    assert int(last["episodes"]) == int(last["episodes_1-1"]) + int(last["episodes_4-1"])
    mean_x = (float(last["x_pos_1-1"]) + float(last["x_pos_4-1"])) / 2
    assert abs(float(last["x_pos"]) - mean_x) < 0.11  # cột chung là trung bình cộng các màn (làm tròn 1 chữ số)

    evals = _rows(run, "evals.csv")
    assert [(r["global_step"], r["level"]) for r in evals] == [("128", "1-1"), ("128", "4-1"),
                                                             ("256", "1-1"), ("256", "4-1")]
    ckpt = sorted((run / "checkpoints").glob("*.pt"))[-1]
    ck = torch.load(ckpt, weights_only=False)
    assert ck["config"]["levels"] == ["1-1", "4-1"] and ck["config"]["level"] == "1-1+4-1"
    for level in ("1-1", "4-1"):  # evaluate.py nguyên bản chấm được checkpoint này trên từng màn
        res = evaluate_checkpoint(ckpt, level=level, episodes=1, seed=2026, max_steps=20)
        assert res["level"] == level and len(res["x_pos"]) == 1
