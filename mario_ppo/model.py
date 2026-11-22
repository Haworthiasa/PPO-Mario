"""Mạng actor-critic cho ảnh Mario.

Giao diện mà train.py và evaluate.py dùng (giữ nguyên tên và chữ ký):
  ActorCritic(n_actions, in_channels=4)
  forward(obs)  : obs uint8 (B, C, 84, 84) -> (logits (B, n_actions) float32, value (B,) float32)
  act(obs)      : lấy mẫu hành động, trả (action long (B,), logp (B,), value (B,)) trên CPU, không gradient
Nhóm được tự chọn kiến trúc (số lớp, kênh, IMPALA-CNN, ...) miễn giữ giao diện trên.
"""
import numpy as np
import torch
from torch import nn


def _init(layer, gain=np.sqrt(2)):
    """Khởi tạo trực giao, bias bằng 0 (thường dùng cho PPO)."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """Thân CNN dùng chung, hai đầu ra: logits của chính sách và giá trị trạng thái.

    Mục đích: một mạng vừa là actor (chọn hành động) vừa là critic (ước lượng giá trị cho GAE).
    Thiết kế:
      - __init__: thân CNN nhận (in_channels, 84, 84), đầu chính sách n_actions logits, đầu giá trị 1 số.
        Khởi tạo đầu chính sách với trọng số rất nhỏ (ví dụ gain 0,01) để chính sách ban đầu gần đều.
      - forward: chuẩn hóa điểm ảnh uint8 về [0, 1], chạy thân, trả (logits, value) với value shape (B,).
    """

    def __init__(self, n_actions, in_channels=4, *, channels=None, hidden_size=None):
        super().__init__()
        use_ltd_defaults = channels is None and hidden_size is None
        first, second, third = (32, 64, 64) if channels is None else channels
        hidden_size = 512 if hidden_size is None else hidden_size

        def init_backbone(layer):
            return _init(layer) if use_ltd_defaults else layer

        encoder = [
            init_backbone(nn.Conv2d(in_channels, first, 8, stride=4)), nn.ReLU(),
            init_backbone(nn.Conv2d(first, second, 4, stride=2)), nn.ReLU(),
            init_backbone(nn.Conv2d(second, third, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
        ]
        projection = [init_backbone(nn.Linear(third * 7 * 7, hidden_size)), nn.ReLU()]
        # Các checkpoint của hai cấu hình dùng tên mô-đun khác nhau.
        if use_ltd_defaults:
            self.trunk = nn.Sequential(*encoder, *projection)
        else:
            self.encoder = nn.Sequential(*encoder)
            self.features = nn.Sequential(*projection)
        self.policy = _init(nn.Linear(hidden_size, n_actions), gain=0.01)
        self.value = _init(nn.Linear(hidden_size, 1), gain=1.0)

    def forward(self, obs):
        if hasattr(self, "trunk"):
            features = self.trunk(obs.float() / 255.0)
        else:
            pixels = obs.to(dtype=self.policy.weight.dtype, memory_format=torch.channels_last)
            features = self.features(self.encoder(pixels / 255.0))
        return self.policy(features), self.value(features).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        """CÓ SẴN. Lấy mẫu từ chính sách hiện tại; nhận obs ở thiết bị bất kỳ, trả tensor CPU."""
        device = next(self.parameters()).device
        logits, value = self(obs.to(device))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.cpu(), dist.log_prob(action).cpu(), value.cpu()
