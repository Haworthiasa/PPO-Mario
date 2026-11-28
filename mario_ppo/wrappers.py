"""Wrapper Gymnasium cho mo-supermario-v0.

Phần CÓ SẴN (không cần sửa): chuyển hành động, sửa lỗi của môi trường gốc, xuất phát ngẫu nhiên, ghi khung hình.
Phần TODO: FrameSkip, GrayResize, ScalarizeReward. Có thể thêm wrapper riêng của nhóm ở cuối tệp
(ví dụ cắt lượt khi Mario đứng yên, thưởng khi chạm cờ, ...) và bật chúng trong env.py.
"""
import cv2
import gymnasium as gym
import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT

# Các tập hành động có thể chọn qua cấu hình env.actions. Hành động 0 luôn là NOOP.
ACTION_SETS = {
    "right_only": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    "simple_down": SIMPLE_MOVEMENT + [["down"]],  # thêm DOWN để chui ống
    "complex": COMPLEX_MOVEMENT,
}

_BUTTON_BITS = {"right": 0x80, "left": 0x40, "down": 0x20, "up": 0x10,
                "start": 0x08, "select": 0x04, "B": 0x02, "A": 0x01, "NOOP": 0x00}


# ---------------------------------------------------------------------------------------------------------
# CÓ SẴN
# ---------------------------------------------------------------------------------------------------------

class ButtonActions(gym.ActionWrapper):
    """Discrete(256) (mặt nạ bit của tay cầm NES) -> Discrete(len(movements))."""

    def __init__(self, env, movements=SIMPLE_MOVEMENT):
        super().__init__(env)
        self._masks = [sum(_BUTTON_BITS[b] for b in combo) for combo in movements]
        self.action_space = gym.spaces.Discrete(len(self._masks))

    def action(self, action):
        return self._masks[int(action)]


class DeathPenaltyFix(gym.Wrapper):
    """Môi trường gốc kết thúc lượt ngay khi Mario bắt đầu chết, trước khi số mạng giảm, nên thành phần
    phần thưởng 'death' luôn bằng 0. Wrapper này trả lại phần phạt -25 (đã co giãn như các thành phần khác).
    Bọc trực tiếp môi trường gốc (trước FrameSkip)."""

    DEATH_INDEX = 2  # thứ tự mục tiêu: [x_pos, time, death, coin, enemy]

    def __init__(self, env):
        super().__init__(env)
        self.penalty = -25.0 * env.unwrapped.reward_space.shape[0] / 150.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated and not info.get("flag_get", False):
            reward = np.array(reward, dtype=np.float32, copy=True)
            if reward[self.DEATH_INDEX] == 0.0:
                reward[self.DEATH_INDEX] = self.penalty
        return obs, reward, terminated, truncated, info


class XPosWrapFix(gym.Wrapper):
    """Tọa độ x 16 bit có thể tràn (lâu đài mê cung đưa Mario về bên trái x = 0), khiến môi trường trả một
    phần thưởng x khổng lồ trong một khung. Bỏ các bước nhảy x không thể có. Bọc môi trường gốc."""

    MAX_DX = 16

    def __init__(self, env):
        super().__init__(env)
        self.scale = env.unwrapped.reward_space.shape[0] / 150.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if abs(reward[0]) > self.MAX_DX * self.scale:
            reward = np.array(reward, dtype=np.float32, copy=True)
            reward[0] = 0.0
        x = info.get("x_pos")
        if x is not None and x >= 0x8000:
            info = dict(info, x_pos=int(x) - 0x10000)
        return obs, reward, terminated, truncated, info


class RandomNoopStart(gym.Wrapper):
    """Mỗi lượt bắt đầu sau một số ngẫu nhiên 0..max_noops bước NOOP, để chính sách không học thuộc một
    nhịp thời gian duy nhất. Script đánh giá dùng cùng ý tưởng với các số NOOP cố định theo seed."""

    NOOP = 0

    def __init__(self, env, max_noops):
        super().__init__(env)
        self.max_noops = max_noops

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(int(self.np_random.integers(0, self.max_noops + 1))):
            obs, _, terminated, truncated, info = self.env.step(self.NOOP)
            if terminated or truncated:
                return self.env.reset()
        return obs, info


class RecordFrames(gym.Wrapper):
    """Giữ mọi khung hình RGB của lượt hiện tại trong self.frames (để ghi video)."""

    def __init__(self, env):
        super().__init__(env)
        self.frames = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs.copy()]
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        self.frames.append(out[0].copy())
        return out


# ---------------------------------------------------------------------------------------------------------
# TODO
# ---------------------------------------------------------------------------------------------------------

class FrameSkip(gym.Wrapper):
    """Lặp lại một hành động `skip` khung hình liên tiếp.

    Mục đích: mỗi quyết định của tác tử kéo dài nhiều khung, giảm số bước cần học và tăng tốc thu dữ liệu.

    step(action) phải:
      - gọi self.env.step(action) tối đa `skip` lần, dừng sớm nếu lượt kết thúc hoặc bị cắt;
      - trả tổng phần thưởng véc-tơ (np.float32, cùng shape với phần thưởng gốc);
      - trả quan sát là max theo từng điểm ảnh của HAI khung cuối (NES nhấp nháy sprite giữa các khung);
        nếu chỉ chạy được một khung thì trả khung đó;
      - trả terminated, truncated, info của khung cuối cùng.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        # Hai khung gần nhất. Phải sao chép vì khung của trình giả lập trỏ thẳng vào bộ nhớ của nó,
        # khung sau ghi đè lên khung trước.
        self._last_two = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)

    def step(self, action):
        total = None
        frames = 0
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            np.copyto(self._last_two[frames % 2], obs)
            frames += 1
            reward = np.asarray(reward, dtype=np.float64)
            total = reward if total is None else total + reward
            if terminated or truncated:
                break
        if frames >= 2:
            obs = np.maximum(self._last_two[0], self._last_two[1])
        else:
            obs = self._last_two[0].copy()
        return obs, total.astype(np.float32), terminated, truncated, info


class GrayResize(gym.ObservationWrapper):
    """Ảnh RGB (H, W, 3) uint8 -> ảnh xám (size, size) uint8.

    Mục đích: giảm kích thước đầu vào của mạng (240x256x3 -> 84x84) mà vẫn giữ đủ thông tin để chơi.
    Ảnh xám phải theo độ sáng (luminance, như cv2.COLOR_RGB2GRAY), không phải trung bình ba kênh.
    Cập nhật self.observation_space cho đúng shape mới.
    """

    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(0, 255, (size, size), np.uint8)
        # Mỗi môi trường chạy trong một tiến trình riêng; để OpenCV tự mở thêm luồng thì các tiến trình
        # tranh lõi CPU với nhau và thu dữ liệu chậm đi.
        cv2.setNumThreads(0)

    def observation(self, obs):
        gray = cv2.cvtColor(np.ascontiguousarray(obs), cv2.COLOR_RGB2GRAY)
        # INTER_AREA lấy trung bình vùng điểm ảnh gốc ứng với mỗi điểm ảnh mới, nên vật nhỏ không bị bỏ sót.
        return cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)


class ScalarizeReward(gym.Wrapper):
    """Phần thưởng véc-tơ 5 mục tiêu [x_pos, time, death, coin, enemy] -> số thực w · r.

    Mục đích: PPO cần phần thưởng vô hướng; trọng số w quyết định tác tử ưu tiên điều gì.
    - __init__: lưu trọng số; ném ValueError nếu số trọng số khác số mục tiêu
      (env.unwrapped.reward_space.shape[0], hoặc 5 nếu môi trường không có reward_space).
    - step: trả phần thưởng kiểu float của Python; giữ véc-tơ gốc trong info["vector_reward"].
    """

    def __init__(self, env, weights):
        super().__init__(env)
        reward_space = getattr(env.unwrapped, "reward_space", None)
        n_objectives = reward_space.shape[0] if reward_space is not None else 5
        self.weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if self.weights.shape[0] != n_objectives:
            raise ValueError(f"cần {n_objectives} trọng số, nhận {self.weights.shape[0]}")

    def step(self, action):
        obs, vector, terminated, truncated, info = self.env.step(action)
        vector = np.asarray(vector, dtype=np.float32)
        info = dict(info, vector_reward=vector)
        return obs, float(np.dot(self.weights, vector)), terminated, truncated, info


# ---------------------------------------------------------------------------------------------------------
# Wrapper bổ sung của nhóm (tùy chọn) - thêm ở đây
# ---------------------------------------------------------------------------------------------------------
# Các wrapper dưới đây chỉ được bật trong khối `if not eval_mode:` của make_env, tức chỉ khi huấn luyện.
# Trong chuỗi wrapper chúng đứng trước ScalarizeReward, nên phần thưởng đi qua chúng còn là véc-tơ.

class NoProgressTruncation(gym.Wrapper):
    """Cắt lượt khi vị trí x xa nhất không tăng trong `max_steps` bước tác tử liên tiếp.

    Mục đích: Mario kẹt trước ống hay đứng yên thì lượt kết thúc sớm, không tốn hàng nghìn bước vô ích.
    Lượt bị cắt báo truncated=True (không phải terminated) để GAE vẫn bootstrap từ quan sát cuối.
    """

    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = int(max_steps)
        self._best_x = None
        self._stalled = 0

    def reset(self, **kwargs):
        self._best_x = None
        self._stalled = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x = int(info.get("x_pos", 0))
        if self._best_x is None or x > self._best_x:
            self._best_x = x
            self._stalled = 0
        else:
            self._stalled += 1
        if self._stalled >= self.max_steps and not terminated:
            truncated = True
            info = dict(info, stalled=True)
        return obs, reward, terminated, truncated, info


class FlagBonus(gym.Wrapper):
    """Cộng `bonus` vào thành phần x_pos của phần thưởng ở bước Mario chạm cờ.

    Mục đích: môi trường gốc không thưởng gì cho việc chạm cờ, nên về đích và chết ngay trước đích gần như
    bằng nhau. Cộng vào thành phần x_pos (trọng số 1 trong cấu hình của nhóm) để "tới đích" là một phần
    của mục tiêu "tiến sang phải".
    """

    X_POS_INDEX = 0

    def __init__(self, env, bonus):
        super().__init__(env)
        self.bonus = float(bonus)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("flag_get", False):
            reward = np.array(reward, dtype=np.float32, copy=True)
            reward[self.X_POS_INDEX] += self.bonus
        return obs, reward, terminated, truncated, info


class PitFallClaw(gym.Wrapper):
    """Khi Mario chết, trừ lại phần thưởng x của quãng đã đi kể từ lần cuối Mario đứng trên nền.

    Mục đích (docs/optimize-1-1.md, Mục 18): nhảy hụt xuống hố thì Mario vẫn trôi sang phải trong lúc rơi, và quãng
    đó được cộng thưởng x như tiến thật. Ở 1-3, vì vậy nhảy hụt được nhiều thưởng hơn cú nhảy đúng lên bệ cao.
    Wrapper này làm cho lượt chết chỉ được tính x tới chỗ Mario rời nền.
    - "Đứng trên nền" đọc từ RAM 0x001D (0 = trên nền, 1 = đang nhảy, 2 = đang rơi, 3 = trượt cột cờ) ở cuối mỗi bước
      tác tử. RAM chỉ dùng để tính phần thưởng lúc huấn luyện; đầu vào của mạng vẫn chỉ là khung hình (đề bài,
      Mục 3.4 và 3.5.2). Dấu hiệu chỉ dựa trên info (y_pos không đổi) sai ở 19/60 lượt chết khi kiểm
      (experiments/pit_claw_check.py), nên không dùng.
    - Lượt chạm cờ hoặc bị cắt (truncated) không bị trừ.
    """

    X_POS_INDEX = 0
    FLOAT_STATE = 0x001D

    def __init__(self, env):
        super().__init__(env)
        self.scale = env.unwrapped.reward_space.shape[0] / 150.0  # cùng hệ số co giãn với thành phần x_pos
        self._ground_x = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ground_x = info.get("x_pos")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x = info.get("x_pos")
        if self._ground_x is None or int(self.env.unwrapped.ram[self.FLOAT_STATE]) == 0:
            self._ground_x = x  # cả ở bước chết: chết khi đang đứng trên nền (chạm địch) thì không trừ gì
        if terminated and not info.get("flag_get", False):
            if x is not None and self._ground_x is not None and x > self._ground_x:
                reward = np.array(reward, dtype=np.float32, copy=True)
                reward[self.X_POS_INDEX] -= (x - self._ground_x) * self.scale
                info = dict(info, pit_fall_claw=int(x - self._ground_x))
        return obs, reward, terminated, truncated, info


class FailureRestart(gym.Wrapper):
    """Một phần lượt huấn luyện bắt đầu lại ngay trước chỗ Mario vừa chết (docs/optimize-1-1.md, Mục 21–23).

    Mục đích: tỉ lệ chết của mỗi vùng giảm theo số lần Mario tới vùng đó, mà các vùng cuối màn được tới ít nhất. Bắt
    đầu lại từ ngay trước chỗ hỏng cho chính sách gặp lại đúng tình huống vừa thua (cùng vị trí cá) mà không phải chạy
    lại cả đoạn đầu màn. Đây là học theo chương trình (đề bài, Mục 3.4); lúc chấm vẫn đi từ đầu màn.

    Trình giả lập của nes_py chỉ có MỘT ô lưu, và ô đó cũng là điểm reset. Vì vậy:
    - Mỗi lượt từ đầu màn chạy trên một chuỗi wrapper (môi trường gốc -> ... -> các mẹo) mà trình giả lập còn đứng ở
      đầu màn. `make_chain()` tạo chuỗi mới; `prepare_restart_chains()` tạo sẵn vài chuỗi, train_joint gọi nó trong
      lúc GPU cập nhật để các môi trường khác không phải chờ.
    - Trong lượt đó, mỗi khi Mario đứng trên nền (RAM 0x001D = 0) ở trạng thái bình thường (RAM 0x000E = 8), đồng hồ
      còn ít nhất `min_time` và bản lưu trước đã cũ từ `backup_every` bước trở lên: lưu trạng thái vào ô lưu của chính
      trình giả lập đó.
    - Lượt kết thúc vì chết (không chạm cờ, không bị cắt) mà bản lưu cũ `min_age`..`max_age` bước: giữ chuỗi đó làm
      một điểm bắt đầu lại, tối đa `pool` điểm, điểm mới thay điểm cũ nhất. Lượt chưa lưu lần nào thì chuỗi vẫn đứng ở
      đầu màn và được dùng lại; các trường hợp khác thì đóng chuỗi.
    - Khi reset: nếu kho có điểm thì với xác suất `prob` bắt đầu lại từ một điểm chọn đều. reset() của chuỗi đó khôi
      phục ô lưu; không thêm NOOP; trong lượt này không lưu gì nên điểm giữ nguyên. Mỗi điểm dùng tối đa `max_uses`
      lần. Còn lại: lượt từ đầu màn với 0..noop_max bước NOOP, như RandomNoopStart.
    - info["restart_episode"] (bool) và info["episode_x0"] (x lúc bắt đầu lượt) có ở mọi bước và ở reset, để
      train_joint tính tỉ lệ chạm cờ chỉ trên lượt từ đầu màn.
    RAM chỉ dùng để chọn lúc lưu; đầu vào của mạng vẫn chỉ là khung hình (đề bài, Mục 3.5.2).
    """

    NOOP = 0
    FLOAT_STATE = 0x001D
    PLAYER_STATE = 0x000E
    PLAYER_NORMAL = 0x08

    def __init__(self, env, make_chain, noop_max=0, prob=0.5, pool=4, max_uses=4, backup_every=16, min_age=6,
                 max_age=48, min_time=100, spares=3):
        super().__init__(env)
        if not (0 <= prob <= 1 and pool >= 1 and max_uses >= 1 and backup_every >= 1 and 0 <= min_age <= max_age
                and spares >= 0):
            raise ValueError("failure_restart: cần 0 <= prob <= 1, pool >= 1, max_uses >= 1, backup_every >= 1, "
                             "0 <= min_age <= max_age, spares >= 0")
        self.make_chain, self.noop_max, self.prob = make_chain, int(noop_max), float(prob)
        self.pool_size, self.max_uses, self.backup_every = int(pool), int(max_uses), int(backup_every)
        self.min_age, self.max_age, self.min_time, self.n_spares = int(min_age), int(max_age), int(min_time), int(spares)
        self._rng = np.random.default_rng()
        self._spares = [env]      # chuỗi đứng ở đầu màn; chuỗi make_env vừa tạo là chuỗi đầu tiên
        self._pool = []           # [chuỗi, số lần đã dùng], cũ nhất trước
        self._entry = None        # mục của kho đang dùng (lượt bắt đầu lại)
        self._restart = False
        self._ended = True
        self._t = 0
        self._backup_step = None  # bước trong lượt của bản lưu gần nhất; None = chưa lưu
        self._backup_time = 0
        self._x0 = 0
        self.created = 0          # số chuỗi đã tạo thêm (cho kiểm tra và nhật ký)

    def _new_chain(self):
        self.created += 1
        return self.make_chain()

    @staticmethod
    def _close(chain):
        try:
            chain.close()
        except Exception:  # noqa: BLE001 - nes_py báo lỗi nếu đóng hai lần
            pass

    def prepare_restart_chains(self):
        """Tạo sẵn chuỗi đứng ở đầu màn cho tới khi đủ `spares`; trả số chuỗi sẵn có."""
        while len(self._spares) < self.n_spares:
            self._spares.append(self._new_chain())
        return len(self._spares)

    def restart_pool_size(self):
        return len(self._pool)

    def _end_episode(self, keep):
        if self._ended:
            return
        self._ended = True
        if self._restart:
            if self._entry[1] >= self.max_uses:
                self._pool.remove(self._entry)
                self._close(self._entry[0])
            return
        if keep:
            self._pool.append([self.env, 0])
            if len(self._pool) > self.pool_size:
                self._close(self._pool.pop(0)[0])
        elif self._backup_step is None:
            self._spares.append(self.env)  # ô lưu vẫn là đầu màn
        else:
            self._close(self.env)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._end_episode(keep=False)  # lượt trước bị bỏ dở (hoặc chưa có lượt nào): không giữ làm điểm
        self._ended, self._t, self._backup_step, self._backup_time = False, 0, None, 0
        if self._pool and self._rng.random() < self.prob:
            self._entry = self._pool[int(self._rng.integers(len(self._pool)))]
            self._entry[1] += 1
            self.env, self._restart = self._entry[0], True
            obs, info = self.env.reset(options=options)
        else:
            self.env = self._spares.pop() if self._spares else self._new_chain()
            self._entry, self._restart = None, False
            obs, info = self.env.reset(seed=seed, options=options)
            for _ in range(int(self._rng.integers(0, self.noop_max + 1))):
                obs, _, terminated, truncated, info = self.env.step(self.NOOP)
                if terminated or truncated:
                    obs, info = self.env.reset()
                    break
        self._x0 = int(getattr(self.env.unwrapped, "_x_position", 0))
        return obs, dict(info, restart_episode=self._restart, episode_x0=self._x0)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        info = dict(info, restart_episode=self._restart, episode_x0=self._x0)
        if terminated or truncated:
            died = terminated and not truncated and not info.get("flag_get", False)
            age = None if self._backup_step is None else self._t - self._backup_step
            self._end_episode(keep=(not self._restart and died and age is not None
                                    and self.min_age <= age <= self.max_age and self._backup_time >= self.min_time))
        elif not self._restart and self._t - (self._backup_step or 0) >= self.backup_every:
            ram = self.env.unwrapped.ram
            time_left = int(info.get("time", 0))
            if (int(ram[self.FLOAT_STATE]) == 0 and int(ram[self.PLAYER_STATE]) == self.PLAYER_NORMAL
                    and time_left >= self.min_time):
                self.env.unwrapped._backup()
                self._backup_step, self._backup_time = self._t, time_left
        return obs, reward, terminated, truncated, info

    def close(self):
        for chain in {id(c): c for c in [self.env, *self._spares, *(e[0] for e in self._pool)]}.values():
            self._close(chain)
