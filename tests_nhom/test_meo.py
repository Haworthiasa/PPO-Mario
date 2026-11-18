"""Kiểm tra của nhóm cho hai mẹo huấn luyện (không thuộc bộ kiểm tra công khai của thầy).

    python -m pytest -q tests_nhom

Dùng trình giả lập thật. Tệp actions_1-1.json là một chuỗi hành động (tập `simple`, frame_skip 4) đưa Mario
tới cờ màn 1-1; trình giả lập tất định nên phát lại chuỗi này luôn tới cờ.
"""
import json
from pathlib import Path

import numpy as np

from mario_ppo.env import make_env

ACTIONS = json.loads((Path(__file__).parent / "actions_1-1.json").read_text())
WEIGHTS = [1.0, 1.0, 2.0, 0.0, 0.0]


def _replay(**kwargs):
    env = make_env("1-1", weights=WEIGHTS, noop_max=0, normalize_reward=False, **kwargs)
    try:
        env.reset(seed=0)
        total, last_reward, info, terminated, truncated = 0.0, 0.0, {}, False, False
        for action in ACTIONS:
            _, last_reward, terminated, truncated, info = env.step(action)
            total += last_reward
            if terminated or truncated:
                break
        return total, last_reward, terminated, truncated, info
    finally:
        env.close()


def test_replay_reaches_flag_and_bonus_is_added_once():
    plain_total, plain_last, terminated, _, info = _replay()
    assert terminated and info["flag_get"]
    bonus_total, bonus_last, terminated, _, info = _replay(flag_bonus=2.0)
    assert terminated and info["flag_get"]
    assert np.isclose(bonus_last - plain_last, 2.0, atol=1e-5)
    assert np.isclose(bonus_total - plain_total, 2.0, atol=1e-4)


def test_tricks_are_off_in_eval_mode():
    _, plain_last, *_ = _replay(eval_mode=True)
    _, eval_last, _, truncated, info = _replay(eval_mode=True, flag_bonus=2.0, max_no_progress_steps=5)
    assert info["flag_get"] and not truncated
    assert np.isclose(eval_last, plain_last, atol=1e-6)


def test_no_progress_truncates_a_standing_mario():
    env = make_env("1-1", weights=WEIGHTS, noop_max=0, normalize_reward=False, max_no_progress_steps=20)
    try:
        env.reset(seed=0)
        steps, terminated, truncated, info = 0, False, False, {}
        while not (terminated or truncated) and steps < 100:
            _, _, terminated, truncated, info = env.step(0)  # NOOP: Mario đứng yên
            steps += 1
        assert truncated and not terminated
        assert info.get("stalled") is True
        assert steps <= 21
    finally:
        env.close()
