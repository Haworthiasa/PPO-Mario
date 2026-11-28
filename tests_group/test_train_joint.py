"""Kiểm tra của nhóm cho mario_ppo/train_joint.py (huấn luyện một mạng trên nhiều màn).

    python -m pytest -q tests_group/test_train_joint.py

Dùng trình giả lập thật, vài lượt huấn luyện tí hon 256 bước (khoảng hai phút).
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


def test_stop_at_step_stops_early_and_resumes_on_the_same_schedule(tmp_path):
    """stop_at_step dừng ở cuối đoạn thu đầu tiên chạm mốc và lưu checkpoint ở đó; tốc độ học vẫn theo total_steps."""
    from mario_ppo.train_joint import train as train_joint
    lr = float(OmegaConf.load("configs/group-4-1.yaml").algo.lr)

    first = train_joint(_tiny_cfg(tmp_path, "stop", levels=["1-1", "4-1"], stop_at_step=100))
    rows = _rows(first)
    assert [int(r["global_step"]) for r in rows] == [64, 128]  # mỗi đoạn thu 64 bước; 128 là bội đầu tiên >= 100
    np.testing.assert_allclose([float(r["lr"]) for r in rows], [lr, lr * 0.75])  # lịch của lượt 256 bước, không co lại
    assert [p.name for p in (first / "checkpoints").glob("*.pt")] == ["step_000000128.pt"]
    ckpt = first / "checkpoints" / "step_000000128.pt"
    assert torch.load(ckpt, weights_only=False)["global_step"] == 128

    second = train_joint(_tiny_cfg(tmp_path, "resume", levels=["1-1", "4-1"], resume=str(ckpt)))
    rows = _rows(second)
    assert [int(r["global_step"]) for r in rows] == [192, 256]  # nối tiếp đúng bộ đếm và chạy hết total_steps
    np.testing.assert_allclose([float(r["lr"]) for r in rows], [lr * 0.5, lr * 0.25])

    with pytest.raises(ValueError):  # mốc dừng phải nằm sau bước của checkpoint được nối tiếp
        train_joint(_tiny_cfg(tmp_path, "bad", levels=["1-1", "4-1"], resume=str(ckpt), stop_at_step=128))


def test_level_shares_follow_failure_rate_with_floor_and_cap():
    from mario_ppo.train_joint import level_shares
    np.testing.assert_allclose(level_shares([1, 1, 1, 1], 0.1, 0.5), [0.25] * 4)  # trượt như nhau: chia đều
    np.testing.assert_allclose(level_shares([0, 0, 0, 0], 0.1, 0.5), [0.25] * 4)  # không màn nào trượt: chia đều
    # sàn 0,1 mỗi màn; 0,6 còn lại chia theo tỉ lệ trượt 3 : 1 : 0 : 0
    np.testing.assert_allclose(level_shares([0.75, 0.25, 0, 0], 0.1, 0.9), [0.55, 0.25, 0.1, 0.1])
    # trần 0,4: màn đầu dừng ở 0,4, phần vượt trần sang màn duy nhất còn trượt
    np.testing.assert_allclose(level_shares([0.75, 0.25, 0, 0], 0.1, 0.4), [0.4, 0.4, 0.1, 0.1])
    # màn duy nhất còn trượt chạm trần: phần thừa chia đều cho các màn không trượt
    np.testing.assert_allclose(level_shares([1, 0, 0, 0], 0.1, 0.4), [0.4, 0.2, 0.2, 0.2])
    for fail in ([0.9, 0.5, 0.1, 0.0, 1.0, 0.3, 0.7, 0.2], [1.0] * 8, [0.0] * 7 + [1.0]):
        share = level_shares(fail, 0.06, 0.30)
        assert abs(share.sum() - 1) < 1e-9 and share.min() >= 0.06 - 1e-9 and share.max() <= 0.30 + 1e-9
    with pytest.raises(ValueError):
        level_shares([1, 1], 0.6, 0.9)     # hai màn, sàn 0,6 mỗi màn: quá 1
    with pytest.raises(ValueError):
        level_shares([1, 1, 1], 0.1, 0.3)  # ba màn, trần 0,3 mỗi màn: không đủ 1


def test_episode_probs_give_long_episodes_fewer_draws():
    from mario_ppo.train_joint import episode_probs
    # cùng phần bước, lượt của màn hai dài gấp ba: màn một phải được chọn gấp ba lần
    np.testing.assert_allclose(episode_probs([0.5, 0.5], [100, 300]), [0.75, 0.25])
    np.testing.assert_allclose(episode_probs([0.5, 0.5], [100, np.nan]), [0.5, 0.5])  # chưa có lượt: dùng độ dài đã biết
    np.testing.assert_allclose(episode_probs([0.2, 0.8], [np.nan, np.nan]), [0.2, 0.8])


def test_vector_env_switches_level_at_episode_end_when_probs_are_set(tmp_path):
    """Qua AsyncVectorEnv thật: đặt level_probs = (0, 1) thì mọi lượt chơi mới là 4-1, lượt đang chơi không đổi."""
    from mario_ppo.train_joint import _make_vec_env
    env_cfg = dict(OmegaConf.to_container(_tiny_cfg(tmp_path, "unused").env), gamma=0.9)
    envs = _make_vec_env(env_cfg, ["1-1", "4-1"], seed=1, switch_levels=["1-1", "4-1"])
    try:
        _, info = envs.reset(seed=1)
        assert list(info["level_index"]) == [0, 1]
        envs.set_attr("level_probs", [(0.0, 1.0)] * 2)
        finished = {0: [], 1: []}  # thế giới của các lượt đã kết thúc, theo môi trường
        for _ in range(40):        # đứng yên: NoProgressTruncation (5 bước) cắt lượt liên tục
            _, _, term, trunc, info = envs.step(np.zeros(2, dtype=np.int64))
            for i in np.flatnonzero(term | trunc):
                fin = info["final_info"]
                finished[int(i)].append(int(fin["world"][i]))
                assert int(fin["level_index"][i]) == {1: 0, 4: 1}[int(fin["world"][i])]
        assert len(finished[0]) >= 3 and finished[0][0] == 1 and set(finished[0][1:]) == {4}
        assert set(finished[1]) == {4}
        assert list(info["level_index"]) == [1, 1]
    finally:
        envs.close()


def test_allocation_run_logs_the_split_and_keeps_progress_columns(tmp_path):
    from mario_ppo.train_joint import train as train_joint
    plain = train_joint(_tiny_cfg(tmp_path, "plain", levels=["1-1", "4-1"]))
    run = train_joint(_tiny_cfg(tmp_path, "alloc", levels=["1-1", "4-1"],
                                allocation={"floor": 0.1, "cap": 0.9, "min_episodes": 1}))
    assert not (plain / "allocation.csv").exists()  # không có mục allocation: không đổi gì so với trước
    assert list(_rows(run)[0]) == list(_rows(plain)[0])  # progress.csv giữ nguyên các cột

    rows = _rows(run, "allocation.csv")
    assert [int(r["global_step"]) for r in rows] == [64, 128, 192, 256]
    assert (rows[0]["steps_1-1"], rows[0]["steps_4-1"]) == ("32", "32")  # đoạn thu đầu: mỗi môi trường một màn
    for r in rows:
        assert int(r["steps_1-1"]) + int(r["steps_4-1"]) == 64  # mỗi bước được ghi cho đúng một màn
        for key in ("share", "prob"):
            assert abs(float(r[f"{key}_1-1"]) + float(r[f"{key}_4-1"]) - 1) < 1e-3
        assert 0.1 - 1e-3 <= float(r["share_1-1"]) <= 0.9 + 1e-3
    last = _rows(run)[-1]
    assert int(last["episodes"]) == int(last["episodes_1-1"]) + int(last["episodes_4-1"]) > 0

    with pytest.raises(ValueError):  # sàn 0,6 cho hai màn là quá 1: báo lỗi trước khi mở môi trường
        train_joint(_tiny_cfg(tmp_path, "bad-alloc", levels=["1-1", "4-1"], allocation={"floor": 0.6, "cap": 0.9}))


def test_kl_controlled_lr_follows_the_band_rule():
    from mario_ppo.train_joint import kl_controlled_lr
    assert kl_controlled_lr(1e-4, 0.05, 0.02, 1.5, 1e-6) == pytest.approx(1e-4 / 1.5)  # KL quá lớn: giảm
    assert kl_controlled_lr(1e-4, 0.005, 0.02, 1.5, 1e-6) == pytest.approx(1.5e-4)    # KL quá nhỏ: tăng
    assert kl_controlled_lr(1e-4, 0.02, 0.02, 1.5, 1e-6) == 1e-4                       # trong dải: giữ
    assert kl_controlled_lr(1e-5, 0.05, 0.02, 1.5, 1e-5) == 1e-5                       # không xuống dưới sàn


def test_lr_control_sets_each_lr_from_the_previous_kl_and_resumes(tmp_path):
    """Mỗi lần cập nhật dùng min(trần giảm tuyến tính, luật KL áp lên tốc độ học vừa dùng); entropy theo bộ đếm bước."""
    from mario_ppo.train_joint import kl_controlled_lr, train as train_joint
    ctl = {"kl_target": 0.02, "factor": 1.5, "lr_max": 4e-4, "lr_min": 1e-5}
    base = OmegaConf.load("configs/group-4-1.yaml").algo
    first = train_joint(_tiny_cfg(tmp_path, "ctl", levels=["1-1"], stop_at_step=128,
                                  algo={"lr_control": ctl, "update_epochs": 4}))
    second = train_joint(_tiny_cfg(tmp_path, "ctl2", levels=["1-1"], algo={"lr_control": ctl, "update_epochs": 4},
                                   resume=str(first / "checkpoints" / "step_000000128.pt")))
    rows = _rows(first) + _rows(second)
    assert [int(r["global_step"]) for r in rows] == [64, 128, 192, 256]
    lr = [float(r["lr"]) for r in rows]
    kl = [float(r["approx_kl"]) for r in rows]
    assert lr[0] == pytest.approx(float(base.lr))
    for i in range(1, 4):
        cap = 4e-4 * (1 - 64 * i / 256)
        # nhật ký làm tròn KL tới 5 chữ số; bỏ qua lần so nếu KL nằm sát mép dải
        if min(abs(kl[i - 1] - 0.03), abs(kl[i - 1] - 0.02 / 1.5)) > 1e-4:
            assert lr[i] == pytest.approx(min(cap, kl_controlled_lr(lr[i - 1], kl[i - 1], 0.02, 1.5, 1e-5)), rel=1e-4)
        assert lr[i] <= cap + 1e-12
    expected_ent = [float(base.ent_coef) + (float(base.ent_coef_final) - float(base.ent_coef)) * s / 256
                    for s in (0, 64, 128, 192)]
    np.testing.assert_allclose([float(r["ent_coef"]) for r in rows], expected_ent, rtol=1e-6)

    with pytest.raises(ValueError):  # lr ban đầu lớn hơn trần
        train_joint(_tiny_cfg(tmp_path, "bad-ctl", levels=["1-1"], algo={"lr_control": dict(ctl, lr_max=1e-5)}))


def test_lr_control_cap_final_keeps_a_floor_under_the_cap(tmp_path):
    """Với cap_final, trần giảm tuyến tính từ lr_max về cap_final thay vì về 0."""
    from mario_ppo.train_joint import kl_controlled_lr, train as train_joint
    ctl = {"kl_target": 0.02, "factor": 1.5, "lr_max": 4e-4, "lr_min": 1e-5, "cap_final": 2e-4}
    run = train_joint(_tiny_cfg(tmp_path, "cap", levels=["1-1"], algo={"lr_control": ctl, "update_epochs": 4}))
    rows = _rows(run)
    lr = [float(r["lr"]) for r in rows]
    kl = [float(r["approx_kl"]) for r in rows]
    for i in range(1, 4):
        cap = 2e-4 + 2e-4 * (1 - 64 * i / 256)
        if min(abs(kl[i - 1] - 0.03), abs(kl[i - 1] - 0.02 / 1.5)) > 1e-4:
            assert lr[i] == pytest.approx(min(cap, kl_controlled_lr(lr[i - 1], kl[i - 1], 0.02, 1.5, 1e-5)), rel=1e-4)
        assert lr[i] <= cap + 1e-12

    with pytest.raises(ValueError):  # trần cuối lớn hơn lr_max
        train_joint(_tiny_cfg(tmp_path, "bad-cap", levels=["1-1"], algo={"lr_control": dict(ctl, cap_final=5e-4)}))
