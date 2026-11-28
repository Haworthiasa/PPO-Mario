"""Kiểm tra của nhóm cho các mẹo huấn luyện (không thuộc bộ kiểm tra công khai của thầy).

    python -m pytest -q tests_group

Dùng trình giả lập thật. Tệp actions_1-1.json là một chuỗi hành động (tập `simple`, frame_skip 4) đưa Mario
tới cờ màn 1-1; trình giả lập tất định nên phát lại chuỗi này luôn tới cờ.
"""
import json
from pathlib import Path

import gymnasium as gym
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
    _, eval_last, _, truncated, info = _replay(eval_mode=True, flag_bonus=2.0, max_no_progress_steps=5,
                                               pit_fall_claw=True)
    assert info["flag_get"] and not truncated
    assert np.isclose(eval_last, plain_last, atol=1e-6)
    assert "pit_fall_claw" not in info


def test_pit_fall_claw_changes_nothing_on_a_run_that_reaches_the_flag():
    plain_total, *_ = _replay()
    claw_total, _, terminated, _, info = _replay(pit_fall_claw=True)
    assert terminated and info["flag_get"]
    assert np.isclose(claw_total, plain_total, atol=1e-5)


class _FakeMario(gym.Env):
    """Môi trường giả: mỗi bước trả (x, float state, kết thúc) theo kịch bản; phần thưởng x = Δx / 30."""

    observation_space = gym.spaces.Box(0, 1, (1,))
    action_space = gym.spaces.Discrete(1)

    def __init__(self, script):
        self.script = script
        self.reward_space = gym.spaces.Box(-1, 1, (5,))
        self.ram = np.zeros(2048, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        self.t, self.x = 0, 40
        self.ram[0x1D] = 0
        return np.zeros(1), {}

    def step(self, action):
        x, fl, done = self.script[self.t]
        self.t += 1
        reward = np.zeros(5, dtype=np.float32)
        reward[0] = (x - self.x) / 30.0
        self.x = x
        self.ram[0x1D] = fl
        return np.zeros(1), reward, done, False, {"x_pos": x, "flag_get": False}


def _run_fake(script):
    from mario_ppo.wrappers import PitFallClaw
    env = PitFallClaw(_FakeMario(script))
    env.reset()
    rewards, info = [], {}
    for _ in script:
        _, reward, terminated, _, info = env.step(0)
        rewards.append(float(reward[0]))
        if terminated:
            break
    return rewards, info


def test_pit_fall_claw_takes_back_x_gained_in_the_air_before_death():
    # chạy trên nền tới x 64, nhảy (float 1), trôi tới x 120 rồi chết khi đang rơi (float 2)
    script = [(52, 0, False), (64, 0, False), (76, 1, False), (90, 1, False), (104, 2, False), (120, 2, True)]
    rewards, info = _run_fake(script)
    assert info["pit_fall_claw"] == 120 - 64
    assert np.isclose(rewards[-1], (120 - 104) / 30.0 - (120 - 64) / 30.0, atol=1e-6)
    assert np.allclose(rewards[:-1], [12 / 30, 12 / 30, 12 / 30, 14 / 30, 14 / 30], atol=1e-6)


def test_pit_fall_claw_ignores_landed_jumps_and_ground_deaths():
    # nhảy rồi đáp ở x 110 (float 0), sau đó chết vì chạm địch khi đang đứng trên nền
    script = [(52, 0, False), (70, 1, False), (90, 1, False), (110, 0, False), (116, 0, True)]
    rewards, info = _run_fake(script)
    assert "pit_fall_claw" not in info
    assert np.isclose(rewards[-1], 6 / 30.0, atol=1e-6)


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


# ---- FailureRestart: bắt đầu lại ngay trước chỗ vừa chết ----

class _FakeNES(gym.Env):
    """Trình giả lập giả có một ô lưu như nes_py: reset() khôi phục ô lưu (ban đầu là đầu màn, x 40).
    Mỗi bước x tăng 12; hành động 1 = đang nhảy (float state 1), 0 = trên nền. Chết khi x >= death_x."""

    observation_space = gym.spaces.Box(0, 1, (1,))
    action_space = gym.spaces.Discrete(2)

    def __init__(self, death_x=None):
        self.ram = np.zeros(2048, dtype=np.uint8)
        self.ram[0x0E] = 0x08
        self.saved = (40, 0)
        self.death_x = death_x
        self.closes = 0

    def _backup(self):
        self.saved = (self.x, self.t)

    @property
    def _x_position(self):
        return self.x

    def reset(self, *, seed=None, options=None):
        self.x, self.t = self.saved
        return np.zeros(1), {}

    def step(self, action):
        self.t += 1
        self.x += 12
        self.ram[0x1D] = 1 if action == 1 else 0
        died = self.death_x is not None and self.x >= self.death_x
        return np.zeros(1), np.zeros(5, np.float32), died, False, {"x_pos": self.x, "flag_get": False, "time": 300}

    def close(self):
        self.closes += 1


def _restart_env(death_x, **kwargs):
    from mario_ppo.wrappers import FailureRestart
    made = []

    def make_chain():
        made.append(_FakeNES(death_x))
        return made[-1]

    first = _FakeNES(death_x)
    return FailureRestart(first, make_chain, noop_max=0, **kwargs), first, made


def _play_until_done(env, action=0, limit=500):
    info = {}
    for _ in range(limit):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info
    raise AssertionError("lượt không kết thúc")


def test_failure_restart_keeps_a_death_point_and_restarts_from_it():
    # chết ở bước 30 (x 400); bản lưu ở bước 16 (x 232), cũ 14 bước: được giữ
    env, first, made = _restart_env(400, prob=1.0, max_uses=2, backup_every=16, min_age=6, max_age=48, spares=0)
    _, info = env.reset(seed=0)
    assert info["restart_episode"] is False and info["episode_x0"] == 40
    _play_until_done(env)
    assert env.restart_pool_size() == 1 and first.saved == (232, 16)
    for use in (1, 2):
        _, info = env.reset()
        assert info["restart_episode"] is True and info["episode_x0"] == 232
        end = _play_until_done(env)
        assert end["restart_episode"] is True and end["x_pos"] == 400
        assert first.saved == (232, 16)  # lượt bắt đầu lại không lưu gì
    assert env.restart_pool_size() == 0 and first.closes == 1  # dùng đủ max_uses lần thì bỏ
    _, info = env.reset()  # kho rỗng: lượt từ đầu màn trên một chuỗi mới
    assert info["restart_episode"] is False and info["episode_x0"] == 40 and len(made) == 1


def test_failure_restart_skips_deaths_right_after_a_backup_and_reuses_clean_chains():
    # chết ở bước 17, ngay sau bản lưu ở bước 16: không giữ, đóng chuỗi
    env, first, made = _restart_env(40 + 12 * 17, prob=1.0, spares=0)
    env.reset(seed=0)
    _play_until_done(env)
    assert env.restart_pool_size() == 0 and first.closes == 1
    # chết ở bước 10, trước bản lưu đầu tiên: chuỗi vẫn đứng ở đầu màn nên được dùng lại
    env, first, made = _restart_env(40 + 12 * 10, prob=1.0, spares=0)
    env.reset(seed=0)
    _play_until_done(env)
    env.reset()
    assert env.restart_pool_size() == 0 and first.closes == 0 and made == [] and env.env is first


def test_failure_restart_does_not_back_up_in_the_air_and_prepares_spares():
    env, first, made = _restart_env(400, prob=1.0, spares=3)
    env.reset(seed=0)
    _play_until_done(env, action=1)  # luôn nhảy: không có bản lưu nào, nên chết mà không giữ điểm
    assert env.restart_pool_size() == 0 and first.saved == (40, 0)
    assert env.prepare_restart_chains() == 3 and len(made) == 2  # chuỗi đầu (còn sạch) + 2 chuỗi mới
    env.close()
    assert first.closes == 1 and all(c.closes == 1 for c in made)


def test_failure_restart_on_the_emulator_restores_the_saved_state():
    from mario_ppo.wrappers import FailureRestart
    cfg = dict(weights=WEIGHTS, noop_max=0, normalize_reward=False,
               failure_restart=dict(prob=1.0, backup_every=16, min_age=0, max_age=400, min_time=0, spares=0))
    env = make_env("1-1", **cfg)
    try:
        restart = env
        while not isinstance(restart, FailureRestart):
            restart = restart.env
        env.reset(seed=0)
        saved, info, done, steps = None, {}, False, 0
        while not done:  # đi bộ sang phải (không nhảy) tới khi Goomba đầu tiên giết Mario
            _, _, terminated, truncated, info = env.step(1)
            done, steps = terminated or truncated, steps + 1
            if not done and restart._backup_step == steps:
                saved = (bytes(env.unwrapped.ram), int(info["x_pos"]), int(info["time"]))
            assert steps < 1000
        assert terminated and not info["flag_get"] and saved is not None
        assert restart.restart_pool_size() == 1
        _, info = env.reset()
        assert info["restart_episode"] is True and info["episode_x0"] == saved[1]
        assert bytes(env.unwrapped.ram) == saved[0]
        _, reward, _, _, info = env.step(0)
        assert 0 <= info["x_pos"] - saved[1] <= 12  # còn đà đi bộ lúc lưu, nên trôi thêm vài điểm ảnh
        # bước đầu sau khi khôi phục: không có thưởng x hay thưởng thời gian khổng lồ (chỉ thưởng x của 1 bước)
        assert -0.2 <= reward <= 0.6
    finally:
        env.close()


def test_failure_restart_is_off_in_eval_mode():
    from mario_ppo.wrappers import FailureRestart
    env = make_env("1-1", weights=WEIGHTS, eval_mode=True, failure_restart=True)
    try:
        layer = env
        while hasattr(layer, "env"):
            assert not isinstance(layer, FailureRestart)
            layer = layer.env
    finally:
        env.close()
