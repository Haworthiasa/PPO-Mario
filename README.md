# mario-ppo: kho mã khởi đầu của đồ án

Đề bài đầy đủ: `de-bai.md`.

## Kết quả của nhóm

Tác tử PPO sau 1,2 triệu bước ở mỗi màn, chấm bằng lệnh ở Mục 4 của đề (20 lượt, seed 2026, lấy mẫu hành động).

| Màn 1-1: chạm cờ 19/20 | Màn 4-1 (màn thưởng): chạm cờ 20/20 |
|:---:|:---:|
| ![Mario vượt màn 1-1](docs/1-1.gif) | ![Mario vượt màn 4-1](docs/4-1.gif) |
| [video gốc](submission/1-1.mp4) · [kết quả](submission/ket-qua-1-1.json) · [cấu hình](configs/group-1-1.yaml) | [video gốc](submission/4-1.mp4) · [kết quả](submission/ket-qua-4-1.json) · [cấu hình](configs/group-4-1.yaml) |

Hai ảnh động được tạo từ video do `evaluate.py` ghi lại (`submission/1-1.mp4`, `submission/4-1.mp4`).

## Cài đặt

Cần Python 3.10–3.12 và trình biên dịch C++ (để cài `nes-py`).

```bash
python -m venv .venv && source .venv/bin/activate      # hoặc: uv venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # bỏ --index-url nếu dùng GPU
pip install -e .
pytest -q          # ban đầu hầu hết bài kiểm tra báo lỗi NotImplementedError
```

Trên Colab: `!pip install -q -e .` trong thư mục dự án.

## Cấu trúc

| Tệp | Vai trò | Cần viết? |
|---|---|---|
| `mario_ppo/wrappers.py` | FrameSkip, GrayResize, ScalarizeReward (+ wrapper riêng của nhóm) | **TODO** |
| `mario_ppo/model.py` | Mạng actor-critic | **TODO** |
| `mario_ppo/ppo.py` | GAE, mất mát PPO, vòng cập nhật | **TODO** |
| `mario_ppo/env.py` | Chuỗi wrapper, chỗ thêm mẹo huấn luyện | có sẵn |
| `mario_ppo/train.py` | Vòng huấn luyện, nhật ký, checkpoint, video | có sẵn |
| `mario_ppo/evaluate.py` | Script đánh giá dùng để chấm | có sẵn, không sửa |
| `configs/default.yaml` | Cấu hình mặc định | sửa tùy ý |
| `tests/` | Bài kiểm tra công khai | |

Gợi ý thứ tự: `test_wrappers` → `test_model` → `test_gae` → `test_ppo` → `test_pipeline`.

```bash
pytest -q tests/test_gae.py          # một nhóm bài kiểm tra
pytest -q -k frameskip               # theo tên
```

## Huấn luyện

```bash
python -m mario_ppo.train                                        # 1-1, cấu hình mặc định
python -m mario_ppo.train level=1-4 algo.total_steps=8000000 seed=2
python -m mario_ppo.train algo.num_envs=8 run_dir=runs/thu-nghiem # máy ít lõi
python -m mario_ppo.train resume=runs/1-1/<run>/checkpoints/step_002000000.pt
```

Mỗi lần chạy ghi vào `runs/<màn>/<thời điểm>/`:

| Tệp | Nội dung |
|---|---|
| `config.yaml` | cấu hình đã dùng |
| `progress.csv` | mỗi lần cập nhật: sps, lr, policy_loss, value_loss, entropy, approx_kl, clipfrac, ep_return, x_pos, flag_rate (100 lượt gần nhất) |
| `evals.csv` | mỗi checkpoint: flag_rate của 5 lượt đánh giá |
| `checkpoints/step_*.pt` | trọng số, optimizer, số bước, cấu hình |
| `videos/step_*.mp4` | lượt tốt nhất ở mỗi checkpoint |

## Đánh giá (giống hệt cách chấm)

```bash
python -m mario_ppo.evaluate --ckpt runs/1-1/<run>/checkpoints/step_003000000.pt --level 1-1 --episodes 20 --seed 2026 --out ket-qua-1-1.json --video 1-1.mp4
```

## Thêm mẹo huấn luyện

1. Viết wrapper mới ở cuối `mario_ppo/wrappers.py`.
2. Bật nó trong khối `if not eval_mode:` của `make_env` (`mario_ppo/env.py`), đọc tham số từ `tricks`.
3. Thêm khóa tương ứng vào mục `env:` của cấu hình, ví dụ `env.max_no_progress_steps=200`.

Mẹo đặt trong khối này không chạy khi đánh giá, nên không ảnh hưởng đến kết quả chấm.

## Phần nhóm thêm

Các tệp có sẵn của thầy (`evaluate.py`, phần `eval_mode` của `env.py`, `tests/`) giữ nguyên. `train.py` chỉ thêm
một khóa tùy chọn (dòng dưới); cấu hình không có khóa đó thì chạy y như bản của thầy.

| Tệp | Nội dung |
|---|---|
| `configs/group-1-1.yaml` | cấu hình của nhóm cho màn 1-1; `configs/default.yaml` là mặc định của thầy, không sửa |
| `configs/group-4-1.yaml` | cấu hình của nhóm cho màn thưởng 4-1, học từ đầu 1,2 triệu bước (lượt chạy `runs/4-1/from-scratch`) |
| `configs/group-4-1-transfer.yaml` | thí nghiệm so sánh: 4-1 khởi tạo từ checkpoint 1-1 (lượt chạy `runs/4-1/transfer`) |
| `mario_ppo/train.py` | thêm khóa `algo.ent_coef_final`: hệ số entropy giảm tuyến tính từ `ent_coef` về giá trị này khi hết `total_steps` |
| `submission/` | checkpoint nộp: `1-1.pt` (từ `runs/1-1/run-1`, bước 1.200.128) và `4-1.pt` (từ `runs/4-1/from-scratch`, bước 1.200.128) |
| `mario_ppo/wrappers.py` (cuối tệp) | hai mẹo huấn luyện: `NoProgressTruncation` (`env.max_no_progress_steps`), `FlagBonus` (`env.flag_bonus`) |
| `mario_ppo/plot.py` | vẽ đồ thị từ `progress.csv` và `evals.csv`; cần thêm `pip install matplotlib` |
| `tests_group/` | kiểm tra hai mẹo trên trình giả lập thật: `python -m pytest -q tests_group` |
| `legacy/` | bản mã nhóm tự viết trước khi có kho mã của thầy, cùng các lần chạy cũ; **không nộp** |

```bash
python -m mario_ppo.train --config configs/group-1-1.yaml          # huấn luyện 1-1 theo cấu hình của nhóm
python -m mario_ppo.train --config configs/group-4-1.yaml run_dir=runs/4-1/from-scratch   # huấn luyện 4-1 từ đầu
python -m mario_ppo.plot runs/1-1/<run>                            # ghi runs/1-1/<run>/curves.png
python -m mario_ppo.evaluate --ckpt <ckpt> --level 1-1 --episodes 100 --seed 7   # đánh giá kỹ để chọn checkpoint
```

Checkpoint trong `legacy/runs/` được huấn luyện với khung hình xử lý khác (không lấy max hai khung cuối), nên
chấm bằng `evaluate.py` của thầy chỉ đạt 0/20 ở màn 1-1. Mọi checkpoint nộp phải huấn luyện lại bằng kho mã này.
