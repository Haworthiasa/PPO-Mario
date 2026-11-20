# Nhật ký thí nghiệm: dạy Mario vượt màn bằng PPO

Tệp này ghi lại theo thứ tự những gì nhóm đã tìm hiểu, đã chạy và đã quyết định, kèm số liệu và lý do. Nó dùng làm nguồn cho báo cáo (Mục 6 của `de-bai.md`) và để thành viên khác hiểu vì sao kho mã có hình dạng hiện tại.

Mọi con số về kết quả trong tệp đều đọc lại từ các tệp trong `runs/`, `submission/` và `legacy/` (không chép theo trí nhớ). Chỗ nào là suy luận hoặc chưa kiểm chứng đều ghi rõ.

## 1. Hiện trạng

| Hạng mục của đề | Điểm | Trạng thái |
|---|---|---|
| A. Cài đặt | 30 | 35/35 bài kiểm tra công khai qua (`pytest -q tests`), thêm 6/6 bài của nhóm (`tests_group`) |
| B. Màn 1-1 | 25 | `submission/1-1.pt`: 19/20 theo lệnh chấm (ngưỡng đủ điểm là 16/20) |
| C. Màn thưởng 4-1 | 15 | `submission/4-1.pt`: 20/20 theo lệnh chấm |
| D. Báo cáo | 25 | chưa viết |
| E. Trình bày | 5 | chưa làm |
| Điểm thưởng (màn ngoài danh sách) | +10 | chưa huấn luyện; đã khảo sát, ứng viên là 2-3 |

Hai checkpoint nộp đều ở bước 1.200.128 của lượt chạy tương ứng:

| Màn | Checkpoint | Cấu hình | Cách khởi tạo | 100 lượt, seed 7 | 20 lượt, seed 2026 (lệnh chấm) |
|---|---|---|---|---|---|
| 1-1 | `runs/1-1/run-1/checkpoints/step_001200128.pt` | `configs/group-1-1.yaml` | từ đầu | 89/100 | 19/20 |
| 4-1 | `runs/4-1/from-scratch/checkpoints/step_001200128.pt` | `configs/group-4-1.yaml` | từ đầu | 96/100 | 20/20 |

Thí nghiệm bổ sung, không nộp: một mạng học cả hai màn (`runs/1-1+4-1/from-scratch`, bước 2.904.064) đạt 99/100 ở 1-1 và 99/100 ở 4-1 (seed 7), 20/20 ở cả hai màn theo lệnh chấm. Chi tiết ở cuối Mục 10.

Tệp `.pt` không nằm trên GitHub (bị `.gitignore` loại); thành viên khác cần nhận riêng.

## 2. Quy ước đọc số liệu

- **Bước** là bước tác tử (một lần mạng chọn hành động, bằng 4 khung hình của trò chơi).
- **Đánh giá 100 lượt, seed 7** dùng để chọn checkpoint và ra quyết định. **Đánh giá 20 lượt, seed 2026** là đúng lệnh chấm ở Mục 4 của đề, chỉ dùng để kiểm tra lần cuối. Cả hai đều lấy mẫu hành động như khi huấn luyện, không dùng mẹo huấn luyện.
- Cờ của 1-1 ở x = 3161, của 4-1 ở x = 3593. Một lượt có x cuối bằng giá trị đó là lượt chạm cờ.
- Mỗi checkpoint trong `evals.csv` chỉ kèm 5 lượt đánh giá. Con số đó dao động quá mạnh để ra quyết định (xem Mục 6).
- Ở lượt chuyển giao, bộ đếm bước nối tiếp từ checkpoint 1-1. "Bước trên 4-1" bằng bộ đếm trừ 1.200.128.

Phần cứng và phần mềm của mọi lượt chạy: AMD Ryzen 5 4600H (6 nhân, 12 luồng), không dùng GPU, WSL2 Ubuntu, Python 3.12.3, torch 2.15.0+cpu, numpy 1.26.4, gymnasium 1.4.0, mo-gymnasium 1.3.2, opencv 4.10.0, omegaconf 2.3.0.

## 3. Đọc đề và dựng môi trường

- Đề lấy từ trang bài tập trên portal; nội dung trùng với `de-bai.md` trong kho của thầy.
- Thử cài trên Windows trước và bỏ: `nes-py` 9 và `gym-super-mario-bros` 9 (bản có sẵn cho Windows) không chạy được với mo-gymnasium 1.3.2. Chuyển sang WSL với bộ cũ hơn (gym-super-mario-bros 7.4.0, nes-py 8.2.1, numpy dưới 2).
- Tốc độ đo được: riêng trình giả lập khoảng 860 bước mỗi giây với 8 môi trường; huấn luyện thật 110–160 bước mỗi giây tùy lượt.

## 4. Bản tự viết trước khi có kho mã của thầy (`legacy/`)

Lúc bắt đầu, kho mã khởi đầu chưa được phát, nên nhóm tự viết toàn bộ `mario_ppo/` theo mô tả trong đề (kể cả `train.py`, `env.py`, `evaluate.py`), có tham khảo hai kho công khai (vietnh1009/Super-mario-bros-PPO-pytorch và yumouwei/super-mario-bros-reinforcement-learning) về cấu hình, không lấy trọng số.

Kết quả của bản này trên 1-1 (một lượt chạy 1.503.232 bước, `legacy/runs/trial-100k/`):

| Bước | 803k | 852k | 901k | 950k | 1,00M | 1,05M | 1,10M | 1,15M | 1,20M | 1,25M | 1,30M | 1,35M | 1,40M | 1,45M | 1,50M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chạm cờ /100 (seed 7) | 87 | 77 | 87 | 83 | 73 | 80 | 84 | 87 | 79 | 85 | 83 | 96 | 85 | 75 | 79 |

Những điều rút ra từ bản này và còn dùng về sau:

- **Cấu hình gốc của nhóm** (8 môi trường × 512 bước, 10 epoch, γ = 0,9, λ = 1,0, ε = 0,15, trọng số phần thưởng [1, 1, 2, 0, 0], thưởng +2 khi chạm cờ, cắt lượt sau 200 bước không tiến) được giữ nguyên khi chuyển sang kho của thầy.
- **Đánh giá 5 lượt không đáng tin.** Có bốn checkpoint liên tiếp đạt 5/5 trong khi đánh giá 100 lượt chỉ 77–87. Từ đó mọi quyết định dùng 100 lượt.
- **Kết quả giữa các checkpoint dao động mạnh** (73 đến 96 trong cùng một lượt chạy). Checkpoint 1,35M được kiểm tra lại với hai seed khác: 85/100 và 95/100.
- **Thất bại do lấy mẫu.** Cùng checkpoint, chọn hành động có xác suất cao nhất thì đạt 100/100; lấy mẫu thì 96 (ở 1,35M) và 79 (ở 1,5M).
- **Chọn 4-1 làm màn thưởng.** Cho checkpoint 1-1 chơi thử 10 màn khác khi chưa học: không màn nào chạm cờ, nhưng 4-1 đi xa gấp đôi các màn còn lại (x trung bình 1128, các màn khác 273–593).

Bản này không nộp được: khi đổi sang định dạng của kho thầy, checkpoint 1,35M chỉ đạt 0/20 theo lệnh chấm (4/100 và 6/100 ở hai lần 100 lượt), vì khung hình đầu vào khác (kho thầy lấy cực đại của hai khung cuối và thu nhỏ bằng OpenCV). Vì vậy 1-1 được huấn luyện lại từ đầu.

## 5. Chuyển sang kho mã của thầy

- Giữ nguyên từng byte các tệp thầy đã viết sẵn: `mario_ppo/evaluate.py`, `mario_ppo/__init__.py`, sáu tệp trong `tests/`, `configs/default.yaml`, `pyproject.toml`, `de-bai.md`. Đã so với bản giải nén mới của tệp zip.
- Điền các TODO trong `wrappers.py`, `model.py`, `ppo.py` theo đúng chữ ký hàm của thầy. Mạng là Nature-CNN (3 lớp tích chập, lớp ẩn 512), 1.688.232 tham số.
- Phần nhóm thêm:
  - `wrappers.py`: hai wrapper chỉ dùng khi huấn luyện, `NoProgressTruncation` (cắt lượt khi không tiến) và `FlagBonus` (thưởng khi chạm cờ).
  - `env.py`: hai dòng gọi chúng, đặt trong khối `if not eval_mode:` như Quy định 3.5 yêu cầu.
  - `ppo.py`: một khóa tùy chọn `algo.ent_coef_final` (lịch giảm hệ số entropy, dùng từ Mục 11). Cấu hình không có khóa này thì hệ số cố định như bản của thầy. Lúc chạy 4-1 và 2-3, 10 dòng của lịch này nằm trong `train.py`; ngày 20/11/2026 chúng được chuyển sang hàm `scheduled_ent_coef` trong `ppo.py` và `train.py` được trả về đúng bản của thầy (trùng mã băm với lần commit đầu). Hàm mới suy số bước từ tốc độ học hiện tại của optimizer nên cần `anneal_lr: true`. Đã đối chiếu: hệ số tính theo cách mới trùng từng bit với công thức cũ ở cả 293 lần cập nhật của lượt 4-1 và 293 lần của lượt 2-3, nên hai checkpoint vẫn tái lập được từ cùng cấu hình.
  - `mario_ppo/plot.py` (vẽ `curves.png`), `tests_group/`, các tệp `configs/group-*.yaml`.
- Đối chiếu Quy định 3.5: mẹo huấn luyện chỉ nằm trong khối cho phép; `info` chỉ dùng để tính phần thưởng và cắt lượt, không đưa vào mạng. Đây là cách nhóm hiểu quy định 2, nên hỏi lại thầy nếu cần chắc chắn.

Xem xét số epoch trước khi chạy (đo trên một lô 8 × 512 thật, hai seed): so với 10 epoch, 4 epoch đạt 68% mức cải thiện mục tiêu chính sách, 6 epoch 84%, 8 epoch 94%; đổi lại tốc độ ước tính 174, 149, 130 so với 116 bước mỗi giây. Tính theo thời gian thì gần như hòa, nên giữ 10 epoch kèm `target_kl` 0,03.

So với cấu hình mặc định của thầy (`configs/default.yaml`): nhóm dùng 8 môi trường × 512 bước thay cho 16 × 256, 10 epoch thay cho 4, lr 1,5e-4 thay cho 2,5e-4, λ 1,0 thay cho 0,95, ε 0,15 thay cho 0,2, hệ số critic 0,5 thay cho 0,25, có `target_kl`, không chuẩn hóa phần thưởng, không tính xu và hạ địch. Nhóm chưa chạy cấu hình mặc định của thầy nên không có số liệu so sánh.

## 6. Huấn luyện màn 1-1 (`runs/1-1/run-1`)

Cấu hình `configs/group-1-1.yaml`, từ đầu, lịch tốc độ học tính cho 2 triệu bước. Lượt chạy gồm hai đoạn: 0 → 700.416 bước, rồi nối tiếp bằng `resume=` tới 1.200.128 bước và dừng ở đó. Tổng thời gian khoảng 2 giờ 35 phút (đoạn đầu 120 bước mỗi giây, đoạn sau 143).

| Bước | 500k | 600k | 700k | 800k | 900k | 1,0M | 1,1M | 1,2M |
|---|---|---|---|---|---|---|---|---|
| Chạm cờ /100 (seed 7) | 46 | 71 | 66 | 83 | 81 | 84 | 80 | **89** |
| Chạm cờ /20 (seed 2026) | – | – | – | 17 | 16 | 19 | 19 | **19** |
| Chạm cờ /5 trong `evals.csv` | 3 | 4 | 2 | 5 | 2 | 5 | 3 | 5 |

Chẩn đoán trong lúc huấn luyện (trung bình theo từng 100 nghìn bước):

- Tỉ lệ chạm cờ khi huấn luyện: 0 trong 200 nghìn bước đầu, 0,24 ở 400–500k, 0,73 ở 600–700k, 0,83–0,89 từ 800k.
- Entropy giảm từ 1,61 xuống 0,69 và đứng yên quanh 0,69–0,71 từ 700k.
- approxKL 0,005 → 0,014, luôn dưới ngưỡng 0,03, gần như không lần nào dừng sớm. Tỉ lệ cắt 0,13 → 0,20–0,24.
- Khi dừng, tốc độ học còn 6,1e-5, tức 40% mức đầu, vì lịch tính cho 2 triệu bước mà chỉ chạy 1,2 triệu.

Checkpoint 1,2M được chọn nộp. Sau đó có thử chạy tiếp lên 1,5M rồi hủy ngay; các dòng ghi thừa sau checkpoint 1,2M đã được cắt khỏi `progress.csv` và `evals.csv`.

## 7. Phân tích chỗ chết ở 1-1

Checkpoint 1,2M, 100 lượt seed 7, 11 lượt thất bại. Ảnh từng trường hợp ở `runs/1-1/run-1/eval/failure-analysis-001200128/`.

| Vị trí x | Số lượt | Chuyện xảy ra (xem từ khung hình) |
|---|---|---|
| 1791–1793 | 3 | hai Goomba ngay sau con Koopa ở x≈1700; Mario đáp xuống ngay trước mặt chúng |
| 1978–2025 | 3 | nhóm Goomba dưới dãy gạch; rơi cạnh một con hoặc nảy từ con này sang con khác |
| 2468–2472 | 3 | hố giữa hai cầu thang; không bật nhảy hoặc nhảy thiếu |
| 2743–2759 | 2 | hai Goomba giữa hai ống cuối, dưới dãy gạch thấp; cú nhảy bị gạch chặn |

- Chọn hành động xác suất cao nhất từ cùng các điểm xuất phát: 97/100. Khoảng 25% hành động lấy mẫu khác với hành động xác suất cao nhất.
- Chỉ 1 trong 11 lượt cứu được nếu sửa 2 bước cuối; đa số đã hỏng từ 6–24 bước trước đó (0,4–1,6 giây).
- Một nguyên nhân cụ thể ở hố: nút nhảy đang được giữ từ trước nên lần bấm "phải + nhảy" tiếp theo không tạo cú nhảy mới (trò chơi cần một lần bấm mới). Mạng không thấy trạng thái nút.

Kết luận: thất bại chủ yếu là nhiễu do lấy mẫu ở những chỗ cần chuỗi hành động chính xác, không phải do chính sách không biết đường.

## 8. Chọn màn thưởng

Cho checkpoint 1-1 (1,2M) chơi thử cả 8 màn trong danh sách khi chưa học (20 lượt, seed 2026). Không màn nào chạm cờ. "Độ khác đầu vào" là trung bình độ lệch tuyệt đối giữa khung quan sát đầu tiên của màn đó và của 1-1 (0 là giống hệt); số này đo bằng một script tạm, không lưu trong kho.

| Màn | Loại nền | Độ khác đầu vào so với 1-1 | x trung bình | x xa nhất |
|---|---|---|---|---|
| **4-1** | ngoài trời, ban ngày | 0,031 | **1157** | **2111** |
| 5-2 | ngoài trời, ban ngày | 0,032 | 546 | 973 |
| 2-1 | ngoài trời, ban ngày | 0,044 | 511 | 1019 |
| 3-1 | ban đêm | 0,365 | 536 | 962 |
| 6-1 | ban đêm | 0,358 | 408 | 645 |
| 3-3 | ban đêm, trên ngọn cây | 0,443 | 322 | 454 |
| 1-4 | lâu đài | 0,345 | 233 | 234 |
| 1-2 | dưới lòng đất | 0,440 | 177 | 965 |

4-1 vừa giống 1-1 nhất về hình ảnh, vừa đi xa gấp đôi màn đứng sau, nên được chọn. Ảnh so sánh các màn: `runs/1-1/run-1/eval/bonus-levels.png`.

## 9. Màn 4-1 bằng học chuyển giao (`runs/4-1/transfer`)

Cấu hình `configs/group-4-1-transfer.yaml`: giống cấu hình 1-1, chỉ đổi màn, nạp checkpoint 1-1 ở 1,2M bằng `resume=` và đặt `total_steps` 6,2 triệu. `resume=` nối tiếp cả trọng số, trạng thái Adam và bộ đếm bước, nên tốc độ học lúc bắt đầu 4-1 là khoảng 1,2e-4.

Lượt chạy gồm ba đoạn nối tiếp nhau:

1. 500 nghìn bước đầu.
2. Một đoạn đặt nhầm mốc dừng (1 triệu bước thay vì 500 nghìn). Nó được dừng tay sau khoảng 640 nghìn bước; giữ lại 600 nghìn bước tới checkpoint gần nhất, bỏ 40 nghìn bước sau đó.
3. 200 nghìn bước cuối.

Tổng cộng giữ lại 1.302.528 bước trên 4-1, khoảng 2 giờ 25 phút (142–162 bước mỗi giây).

| Bước trên 4-1 | 100k | 200k | 300k | 400k | 500k | 600k | 700k | 800k | 900k | 1,0M | 1,1M | 1,2M | 1,3M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chạm cờ /100 (seed 7) | 1 | 10 | 31 | 54 | 60 | 56 | 80 | 71 | 84 | 82 | 83 | 74 | 81 |
| Chạm cờ /20 (seed 2026) | 2 | 1 | 9 | 7 | 9 | 13 | 13 | 14 | 15 | 15 | 16 | 17 | 17 |

Chẩn đoán:

- Năm checkpoint cuối dao động 74–84/100, không lên thêm.
- approxKL khoảng 0,022 ở mọi đoạn, và mỗi lần cập nhật chỉ chạy 117–144 trên 160 bước gradient vì chạm ngưỡng `target_kl` 0,03. Tỉ lệ cắt 0,24–0,29. Chính sách bị đẩy mạnh hơn hẳn so với lượt 1-1.
- Trong 400 nghìn bước cuối, entropy đứng quanh 0,61–0,64 và tỉ lệ chạm cờ khi huấn luyện đứng ở 0,82–0,85.
- Tốc độ học chỉ giảm từ 1,2e-4 xuống 0,9e-4.

Phân tích chỗ chết ở checkpoint cuối (100 lượt seed 7, 19 lượt thất bại; ảnh ở `runs/4-1/transfer/eval/failure-analysis-002502656/`): 7 lượt đáp trúng cây ăn thịt trên ống, 7 lượt chết vì gai của Lakitu (hai lượt ở x≈1842 do kẹt trước một ống cao vì hành động lấy mẫu nhả nút nhảy sớm), 5 lượt rơi hố. Chọn hành động xác suất cao nhất: 95/100, so với 81/100 khi lấy mẫu.

## 10. Chuyển giao làm quên màn cũ

Câu hỏi đặt ra: sau khi học 4-1, checkpoint có còn chơi được 1-1 không, để nộp một mạng cho cả hai màn?

Cho mọi checkpoint của lượt chuyển giao chơi 1-1 (20 lượt, seed 2026; kết quả ở `runs/4-1/transfer/eval/eval-on-1-1/`):

| Bước trên 4-1 | 0 | 100k | 200k | 300k | 400k | 500k | 600k | 700k | 800k | 900k | 1,0M | 1,1M | 1,2M | 1,3M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chạm cờ ở 1-1 /20 | 19 | 10 | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 |
| Chạm cờ ở 4-1 /20 | 0 | 2 | 1 | 9 | 7 | 9 | 13 | 13 | 14 | 15 | 15 | 16 | 17 | 17 |

- Mạng quên 1-1 trong 100–200 nghìn bước đầu, trước cả khi học được 4-1. Nguyên nhân là dữ liệu huấn luyện chỉ còn 4-1, không phải do chạy ít bước.
- Một checkpoint không dùng được cho cả hai màn. Đề cũng yêu cầu nộp checkpoint riêng cho từng màn (Mục 7), nên điều này không mất điểm.
- Một mạng cho nhiều màn cần huấn luyện đồng thời trên các màn đó và không được thêm điểm. LoRA (đóng băng mạng gốc, thêm lớp nhỏ cho từng màn) cũng tránh được quên nhưng không đem lại điểm, nên không làm.

### Tiếp nối (20/11/2026): mã cho một mạng học hai màn cùng lúc

Ý tưởng của nhóm: một mạng chơi được cả 1-1 và 4-1, tổng 3 triệu bước. Đây là thí nghiệm bổ sung cho báo cáo, không thay hai tệp nộp và không cộng điểm B, C.

- Số liệu chéo mới đo: mạng chuyên 4-1 chơi 1-1 được 0/20, 19 lượt dừng ở x ≈ 680–713 (`runs/4-1/from-scratch/eval/eval-on-1-1/`). Chiều ngược lại là 0/20 (Mục 8). Hai màn không cho nhau kỹ năng sẵn.
- Không luân phiên theo thời gian: bảng trên cho thấy 100–200 nghìn bước chỉ học một màn là đủ làm hỏng màn kia. Thay vào đó trộn hai màn trong từng lần cập nhật: 8 môi trường, 4 cái chơi 1-1 và 4 cái chơi 4-1.
- Thang lợi tức hai màn gần nhau (khoảng 93–96 ở 1-1 và 110–116 ở 4-1 khi đã giỏi, lượt dài 260 và 295 bước), nên dùng chung chuẩn hóa lợi thế, không chuẩn hóa riêng từng màn.
- Mã: `mario_ppo/train_joint.py` (tệp mới của nhóm; `train.py` của thầy không đổi). Vòng lặp chép theo `train.py`, khác ba chỗ: gán màn cho từng môi trường theo khóa `levels`, ghi nhật ký riêng từng màn, đánh giá từng màn ở mỗi checkpoint. Mạng không được báo đang ở màn nào; `info` chỉ dùng để ghi nhật ký và tự kiểm tra môi trường chơi đúng màn đã gán.
- Cấu hình: `configs/group-1-1-and-4-1.yaml`, giống `configs/group-4-1.yaml`, chỉ khác `levels` và `total_steps` 3.000.000.
- Kiểm tra (`tests_group/test_train_joint.py`, 3 bài): với một màn, `train_joint.py` cho đúng các con số và trọng số của `train.py`; với hai màn, nhật ký và đánh giá có đủ từng màn và `evaluate.py` nguyên bản chấm được checkpoint trên từng màn.
- Chạy thử 12.288 bước với cấu hình thật: khoảng 145 bước mỗi giây, ghi đủ checkpoint, video và số liệu từng màn. `mario_ppo/plot.py` vẽ lần chạy nhiều màn với mỗi màn một màu.
- Lượt 3 triệu bước đã chạy sau khi viết đoạn này; kết quả ở mục con ngay dưới.
- Giới hạn khi đọc kết quả: mạng chung khác mạng chuyên 1-1 ở cả cấu hình (1-1 dùng tốc độ học 1,5e-4, không có lịch entropy) lẫn số bước mỗi màn (1,5 triệu so với 1,2 triệu), nên không phải thí nghiệm "đổi đúng một lựa chọn" của Mục 6.4 trong đề.

### Kết quả lượt 3 triệu bước (`runs/1-1+4-1/from-scratch`)

Chạy một mạch từ đầu, seed 1, 3.002.368 bước (1,5 triệu bước mỗi màn), 6 giờ 6 phút, 30 checkpoint. Mỗi checkpoint được chấm trên từng màn bằng `evaluate.py` nguyên bản với `--level` (kết quả ở `runs/1-1+4-1/from-scratch/eval/<màn>/`). Cột "màn yếu hơn" là số nhỏ hơn trong hai màn, dùng để chọn checkpoint.

| Bước | 1-1 /100 (seed 7) | 4-1 /100 (seed 7) | Màn yếu hơn /100 | 1-1 /20 (seed 2026) | 4-1 /20 (seed 2026) |
|---|---|---|---|---|---|
| 102.400 | 0 | 0 | 0 | 0 | 0 |
| 200.704 | 0 | 0 | 0 | 0 | 0 |
| 303.104 | 4 | 0 | 0 | 1 | 0 |
| 401.408 | 7 | 3 | 3 | 1 | 0 |
| 503.808 | 22 | 9 | 9 | 2 | 1 |
| 602.112 | 29 | 7 | 7 | 2 | 2 |
| 700.416 | 28 | 16 | 16 | 6 | 2 |
| 802.816 | 56 | 20 | 20 | 9 | 5 |
| 901.120 | 64 | 34 | 34 | 13 | 9 |
| 1.003.520 | 73 | 40 | 40 | 15 | 8 |
| 1.101.824 | 78 | 55 | 55 | 15 | 11 |
| 1.200.128 | 92 | 71 | 71 | 16 | 14 |
| 1.302.528 | 82 | 66 | 66 | 18 | 16 |
| 1.400.832 | 79 | 84 | 79 | 16 | 18 |
| 1.503.232 | 88 | 84 | 84 | 15 | 16 |
| 1.601.536 | 85 | 88 | 85 | 18 | 18 |
| 1.703.936 | 92 | 87 | 87 | 19 | 18 |
| 1.802.240 | 90 | 88 | 88 | 19 | 18 |
| 1.900.544 | 94 | 94 | 94 | 18 | 16 |
| 2.002.944 | 92 | 88 | 88 | 18 | 18 |
| 2.101.248 | 91 | 94 | 91 | 19 | 19 |
| 2.203.648 | 96 | 96 | 96 | 19 | 19 |
| 2.301.952 | 93 | 95 | 93 | 20 | 20 |
| 2.400.256 | 93 | 96 | 93 | 18 | 20 |
| 2.502.656 | 97 | 97 | 97 | 20 | 20 |
| 2.600.960 | 98 | 98 | 98 | 17 | 18 |
| 2.703.360 | 97 | 99 | 97 | 19 | 20 |
| 2.801.664 | 96 | 100 | 96 | 20 | 20 |
| **2.904.064** | **99** | **99** | **99** | **20** | **20** |
| 3.002.368 | 100 | 99 | 99 | 18 | 20 |

- **Checkpoint được chọn: bước 2.904.064.** Nó và checkpoint cuối cùng có màn yếu hơn đạt 99/100; chọn nó vì ở seed 2026 nó đạt 20/20 cả hai màn, còn checkpoint cuối trượt 2 lượt ở 1-1 (x = 679 và 1792). Hai lượt thất bại của nó trong 200 lượt seed 7: x = 680 ở 1-1 và x = 2406 ở 4-1. Chưa xem khung hình các lượt này.
- **Một mạng giữ được cả hai màn.** So với bảng quên ở đầu Mục 10 (1-1 rơi từ 19/20 xuống 2/20 sau 200 nghìn bước chỉ học 4-1), trộn hai màn trong từng lần cập nhật không làm màn nào tụt.
- **So với hai mạng chuyên đang nộp** (1-1: 89/100 và 19/20; 4-1: 96/100 và 20/20): mạng chung cao hơn ở cả hai màn. Ở cùng số bước mỗi màn (bước 2.400.256, tức 1,2 triệu bước mỗi màn) mạng chung đạt 93 và 96, so với 89 và 96. Học chung không làm chậm màn nào tính theo số bước của riêng màn đó; phần hơn còn lại trùng với việc chạy dài hơn.
- **Chẩn đoán.** Entropy trung bình 200 nghìn bước đầu là 1,52, 200 nghìn bước cuối là 0,30. Từ 400 nghìn đến 1,8 triệu bước approxKL đứng quanh 0,021 và `target_kl` cắt bớt bước gradient (thấp nhất trung bình 117/160 lần cập nhật minibatch mỗi vòng ở 800 nghìn–1 triệu bước); từ 2,2 triệu bước không còn lần dừng sớm nào và approxKL giảm về 0,002 theo tốc độ học. Phương sai được giải thích tăng từ 0,54 lên 0,93 (cũng là trung bình 200 nghìn bước đầu và cuối). Tỉ lệ chạm cờ khi huấn luyện lúc kết thúc: 0,99 ở 1-1 và 1,00 ở 4-1.
- **Nếu dừng ở nửa lượt** (1,5 triệu bước) thì checkpoint tốt nhất khi đó là 88 và 84 trên 100, và 15/20 ở 1-1 với seed 2026. Mọi checkpoint từ 2,5 triệu bước đều đạt ít nhất 96/100 ở cả hai màn. Điều này khớp với điều 2 của Mục 14.
- **Giới hạn.** Mỗi bên chỉ có một lượt chạy với một seed, nên chênh lệch vài điểm phần trăm chưa đủ để kết luận; thêm các khác biệt về cấu hình đã nêu ở cuối mục con trước.
- **Trạng thái nộp.** `submission/` chưa đổi: hai tệp nộp vẫn là hai mạng chuyên. Checkpoint của mạng chung lưu cấu hình với `level: "1-1+4-1"` (chỉ là nhãn), nên khi chấm phải truyền `--level`, đúng như lệnh ở Mục 4 của đề.

## 11. Màn 4-1 học từ đầu (`runs/4-1/from-scratch`) — bản nộp

Lý do thử: lượt chuyển giao tốn 1,2M bước ở 1-1 rồi 1,3M bước ở 4-1 mà vẫn dừng quanh 81%. Cấu hình `configs/group-4-1.yaml` rút từ chẩn đoán của hai lượt trước, khác cấu hình 1-1 ở ba điểm:

| Thay đổi | Giá trị | Lý do |
|---|---|---|
| `total_steps` | 1.200.000, bằng đúng số bước chạy | để tốc độ học giảm về 0 ở cuối lượt; hai lượt trước dừng khi tốc độ học còn 40–60% và kết quả còn dao động mạnh giữa các checkpoint |
| `lr` | 2,0e-4 thay cho 1,5e-4 | giữ tốc độ học trung bình cả lượt quanh 1,0e-4 dù lịch giảm nhanh hơn; đầu lượt 1-1 approxKL chỉ 0,005 nên còn chỗ cho bước dài hơn |
| `ent_coef_final` (khóa mới của nhóm; lúc chạy nằm trong `train.py`, nay ở `ppo.py`, xem Mục 5) | hệ số entropy giảm tuyến tính 0,01 → 0,002 | đề chấm bằng lấy mẫu; ở hai lượt trước entropy đứng yên cùng lúc tỉ lệ chạm cờ đứng yên, và chọn hành động xác suất cao nhất luôn tốt hơn lấy mẫu |

Chạy một mạch 1.200.128 bước, 2 giờ 59 phút, trung bình 112 bước mỗi giây. Đầu lượt chỉ được 83 bước mỗi giây, thấp hơn mức 109–120 của lượt 1-1 ở cùng giai đoạn; nguyên nhân chưa xác định, có thể do máy bận việc khác. Lượt chuyển giao nhanh hơn cả hai (142–162) vì các lần dừng sớm cắt bớt bước gradient.

| Bước | 100k | 200k | 300k | 400k | 500k | 600k | 700k | 800k | 900k | 1,0M | 1,1M | 1,2M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chạm cờ /100 (seed 7) | 0 | 2 | 4 | 14 | 22 | 53 | 55 | 84 | 81 | 90 | 98 | **96** |
| Chạm cờ /20 (seed 2026) | 0 | 1 | 3 | 2 | 5 | 10 | 12 | 15 | 17 | 19 | 19 | **20** |

Chẩn đoán:

- Entropy giảm đều từ 1,61 xuống 0,52, không đứng yên ở 0,6–0,7 như hai lượt trước.
- approxKL lên cao nhất 0,0145 ở 400–600k rồi giảm về 0,002 cuối lượt; tỉ lệ cắt từ 0,26 về 0,05. Chính sách ổn định dần theo tốc độ học.
- Tỉ lệ chạm cờ khi huấn luyện 0,95 trong 200 nghìn bước cuối.
- Bốn lượt thất bại ở checkpoint cuối (100 lượt): x = 483, 2409, 2838, 2850. Chưa xem khung hình.

So với lượt chuyển giao ở cùng số bước trên 4-1: chậm hơn trong 500 nghìn bước đầu (22 so với 60 trên 100), ngang nhau ở 800–900 nghìn, vượt hẳn từ 1 triệu (90, 98, 96 so với 82, 83, 74).

**Giới hạn của kết luận.** Lượt này khác lượt chuyển giao ở bốn thứ cùng lúc (cách khởi tạo, tốc độ học, độ dài lịch, lịch entropy) và mỗi bên chỉ chạy một seed. Vì vậy chưa biết thay đổi nào đóng góp bao nhiêu, và chưa thể coi đây là thí nghiệm so sánh đúng nghĩa của Mục 6.4 trong đề.

## 12. Tệp nộp và GitHub

- Tạo bằng đúng lệnh ở Mục 4 của đề, có `--out` và `--video`: `submission/ket-qua-1-1.json` (19/20, lượt hỏng dừng ở x = 1673), `submission/1-1.mp4`, `submission/ket-qua-4-1.json` (20/20), `submission/4-1.mp4`. Vị trí x từng lượt trùng với lần đánh giá 20 lượt trước đó trên cùng checkpoint. Máy của thầy có thể cho kết quả lệch nhẹ; chưa kiểm tra trên máy khác.
- `submission/1-1.pt` và `submission/4-1.pt` là bản sao của hai checkpoint nộp.
- Nhánh `daylabranchcuaDung` trên kho của nhóm đã có mã, cấu hình, kết quả 1-1 và 4-1, README kèm hai ảnh động (`docs/1-1.gif`, `docs/4-1.gif`). Các tệp JSON của kết quả khảo sát ở Mục 8, Mục 13 và của đường cong quên ở Mục 10 cũng đã có trên nhánh; checkpoint và video huấn luyện thì không.

Tên thư mục và tệp đã được đổi sang tiếng Anh sau khi chạy xong. Tệp `config.yaml` và các tệp JSON trong `runs/` là bản ghi của lúc chạy nên bên trong vẫn ghi tên cũ; bảng dưới đây để tra ngược.

| Tên lúc chạy | Tên hiện tại |
|---|---|
| `runs/1-1/lan-1` | `runs/1-1/run-1` |
| `runs/4-1/lan-1` (sau đó là `chuyen-giao`) | `runs/4-1/transfer` |
| `runs/4-1/lan-2` (sau đó là `tu-dau`) | `runs/4-1/from-scratch` |
| `danh-gia/`, `thu-<màn>/`, `phan-tich-<bước>/` | `eval/`, `eval-on-<màn>/`, `failure-analysis-<bước>/` |
| `nop/` | `submission/` |
| `configs/nhom-*.yaml` | `configs/group-*.yaml` |
| `tests_nhom/` | `tests_group/` |
| `ban-cu/` | `legacy/` |

Giữ nguyên tên: `de-bai.md` (tệp của thầy), `ket-qua-1-1.json` và `ket-qua-4-1.json` (tên trong lệnh chấm của đề).

## 13. Khảo sát màn ngoài danh sách (cho điểm thưởng +10)

Cho hai checkpoint nộp chơi thử 23 màn ngoài danh sách khi chưa học (20 lượt, seed 2026). Không màn nào chạm cờ. Kết quả ở `runs/1-1/run-1/eval/eval-on-<màn>/` và `runs/4-1/from-scratch/eval/eval-on-<màn>/`.

| Màn | Checkpoint 1-1: x trung bình / xa nhất | Checkpoint 4-1: x trung bình / xa nhất |
|---|---|---|
| 2-2 | 1096 / 1618 | 822 / 2163 |
| **2-3** | 843 / 1762 | 1024 / 1792 |
| 8-3 | 601 / 1015 | 923 / 1237 |
| 7-2 | 810 / 1235 | 487 / 1109 |
| 6-2 | 318 / 448 | 686 / 1278 |
| 7-3 | 637 / 1080 | 582 / 891 |
| 7-1 | 559 / 882 | 524 / 985 |
| 4-3 | 508 / 578 | 466 / 710 |
| 5-1 | 403 / 940 | 493 / 807 |
| 8-1 | 401 / 1074 | 461 / 1072 |
| 1-3 | 416 / 445 | 403 / 747 |
| 8-2 | 305 / 1428 | 408 / 1438 |
| 3-2 | 425 / 1107 | 264 / 660 |
| 6-3 | 299 / 373 | 350 / 480 |
| 5-3 | 360 / 702 | 267 / 418 |
| 4-2 | 281 / 466 | 249 / 350 |
| 7-4 | 301 / 358 | 340 / 347 |
| 5-4 | 295 / 378 | 337 / 347 |
| 2-4 | 291 / 344 | 337 / 347 |
| 3-4 | 324 / 521 | 270 / 274 |
| 6-4 | 233 / 234 | 233 / 235 |
| 8-4 | 147 / 147 | 238 / 1159 |
| 4-4 | 123 / 133 | 108 / 147 |

Đề xuất hiện tại là **2-3**: cả hai checkpoint đều đi xa (tới x≈1760–1790), lượt chơi ngắn, và các lượt chết dồn ở vài mốc rõ (x≈1079, 1335, 1571, 1790). 2-2 đứng thứ hai nhưng mỗi lượt dài gấp 2–4 lần về số bước. Tám màn lâu đài (x-4) đều kém.

Lưu ý: tín hiệu lần này yếu hơn lần chọn 4-1, vì không màn nào vượt hẳn các màn còn lại. Nhận xét "2-2 là màn dưới nước, 2-3 là màn cầu có cá nhảy" đến từ hiểu biết chung về trò chơi, chưa mở khung hình ra kiểm tra. Chưa huấn luyện màn nào trong số này.

## 14. Những điều đã học được

1. **Ra quyết định bằng 100 lượt, không phải 5 hay 20.** Hai checkpoint cạnh nhau có thể chênh 10–20 điểm phần trăm, và 5 lượt trong `evals.csv` từng cho 2/5 ở checkpoint đạt 81/100.
2. **Cho tốc độ học về 0 ở cuối lượt.** Hai lượt dừng giữa lịch đều kết thúc với kết quả dao động; lượt duy nhất chạy hết lịch kết thúc ổn định ở 96–98/100. Lượt một mạng hai màn (cuối Mục 10) cũng chạy hết lịch và kết thúc ở 96–100/100 trên cả hai màn. Đây là quan sát trên bốn lượt, chưa phải thí nghiệm tách riêng.
3. **Đề chấm bằng lấy mẫu, nên độ ngẫu nhiên của chính sách ở cuối lượt quyết định điểm.** Ở cả bốn checkpoint đã so sánh (hai của bản cũ, 1-1 và 4-1 chuyển giao của bản mới), chọn hành động xác suất cao nhất đạt 95–100/100, còn lấy mẫu chỉ 79–96.
4. **Học chuyển giao giúp nhanh lúc đầu nhưng không giúp về đích.** Ở 4-1 nó nhanh hơn trong 500 nghìn bước đầu rồi dừng quanh 81%, đồng thời xóa khả năng chơi 1-1.
5. **Quãng đường đi được khi chưa học là cách rẻ để chọn màn.** Nó đã chọn đúng 4-1; mỗi lần khảo sát cả danh sách chỉ mất vài phút.
6. **Ghi rõ bộ đếm bước đang nói tới.** "2,5M" ở lượt chuyển giao là 1,2M ở 1-1 cộng 1,3M ở 4-1; nhầm chỗ này đã gây hiểu lầm nhiều lần.
7. **Chỉ chạy đúng số bước đã thống nhất.** Một lần đặt nhầm mốc dừng làm lượt chuyển giao có một đoạn phải cắt bỏ.
8. **Muốn một mạng chơi nhiều màn thì trộn các màn trong từng lần cập nhật.** Học lần lượt làm quên màn cũ sau 100–200 nghìn bước; chia đôi 8 môi trường cho hai màn thì mạng đạt 99/100 ở cả hai (Mục 10). Mới thử với một seed.

Sự cố kỹ thuật đáng nhớ:

- Thu nhỏ ảnh chậm gấp 10 lần khi thư viện BLAS chạy đa luồng (bản tự viết); kho của thầy dùng OpenCV nên không gặp.
- `evaluate.py` không giới hạn số luồng: chạy nhiều lần đánh giá song song phải đặt `OMP_NUM_THREADS` từ bên ngoài, nếu không chúng tranh nhau 12 luồng.
- Máy ngủ thì lượt chạy tạm dừng rồi chạy tiếp khi máy thức; kết quả không hỏng nhưng thời gian đo sai.
- Dừng huấn luyện giữa chừng để lại các dòng thừa trong `progress.csv` và `evals.csv` sau checkpoint cuối; phải cắt chúng trước khi `resume=`.

## 15. Việc còn lại

Cho báo cáo (Mục 6 của đề), so với những gì đã có:

| Mục báo cáo | Đã có | Còn thiếu |
|---|---|---|
| 1. Cài đặt, bảng `pytest` | mã, 35/35 và 6/6 bài qua | viết mô tả |
| 2. Thiết lập huấn luyện | cấu hình, phần cứng, số bước, thời gian, tốc độ (Mục 2, 6, 9, 11 ở trên) | lập bảng |
| 3. Đường cong học | `curves.png` của ba lượt chạy, `progress.csv` | viết nhận xét cho từng hình |
| 4. Thí nghiệm so sánh | lượt chuyển giao so với lượt từ đầu, nhưng khác nhau bốn thứ | một thí nghiệm chỉ đổi một lựa chọn, ví dụ cấu hình 4-1 bỏ `ent_coef_final` |
| 5. Phân tích thất bại | bảng chỗ chết và ảnh cho 1-1 và 4-1 chuyển giao | biểu đồ vị trí x cuối lượt của hai checkpoint nộp; xem khung hình 4 lượt hỏng của checkpoint 4-1 nộp |
| 6. Phân công | – | cả nhóm điền |

Ngoài báo cáo: huấn luyện màn ngoài danh sách (nếu nhóm muốn lấy +10), tệp trình chiếu, đóng gói `nhom-<số>.zip` theo Mục 7 của đề.

Câu hỏi còn để ngỏ cho thầy: nộp mã mốc tuần 1 ở đâu; nộp tệp zip thế nào khi ô nộp trên portal là ô nhập chữ; dùng `info` để tính phần thưởng và cắt lượt khi huấn luyện có đúng với Quy định 3.5 không.

## 16. Tìm tệp ở đâu

| Cần gì | Ở đâu |
|---|---|
| Cấu hình nộp | `configs/group-1-1.yaml`, `configs/group-4-1.yaml` |
| Tệp nộp theo Mục 4 của đề | `submission/` |
| Số liệu huấn luyện từng lượt | `runs/<màn>/<lượt>/progress.csv`, `evals.csv`, `config.yaml`, `curves.png` |
| Đánh giá từng checkpoint | `runs/<màn>/<lượt>/eval/100-seed7_<bước>.json`, `20-seed2026_<bước>.json` |
| Chơi thử màn khác | `runs/<màn>/<lượt>/eval/eval-on-<màn khác>/` |
| Lượt một mạng hai màn | `runs/1-1+4-1/from-scratch/`; đánh giá từng màn ở `eval/1-1/` và `eval/4-1/`; mã `mario_ppo/train_joint.py`, cấu hình `configs/group-1-1-and-4-1.yaml` |
| Ảnh phân tích chỗ chết | `runs/1-1/run-1/eval/failure-analysis-001200128/`, `runs/4-1/transfer/eval/failure-analysis-002502656/` |
| Bản tự viết cũ, không nộp | `legacy/` (không có trên GitHub) |

Lệnh dùng lại:

```bash
python -m mario_ppo.train --config configs/group-4-1.yaml run_dir=runs/4-1/from-scratch     # huấn luyện
python -m mario_ppo.evaluate --ckpt <checkpoint.pt> --level 4-1 --episodes 100 --seed 7 --out <tệp.json>   # chọn checkpoint
python -m mario_ppo.plot runs/4-1/from-scratch                                              # vẽ lại curves.png
```
