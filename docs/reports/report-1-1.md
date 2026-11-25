# Báo cáo màn 1-1

Cập nhật lần cuối: 24/11/2026, 10:30. Lượt 8 màn đã kết thúc ở bước 9.998.336; 10 checkpoint của nó đã được chấm 100 lượt seed 7 trên 1-1, 2 checkpoint đã được chấm theo lệnh của đề. Checkpoint cuối đã được chấm thêm 100 lượt seed 11.

Quy ước: **bước** là bước tác tử. **Số chấm** là kết quả của `evaluate.py` (100 lượt seed 7 để ra quyết định, 20 lượt seed 2026 là đúng lệnh chấm của đề). **Số lúc huấn luyện** là tỉ lệ chạm cờ của 100 lượt huấn luyện gần nhất trong `progress.csv`; hai loại số có thể lệch nhau hơn 10 điểm. Cờ của 1-1 ở x = 3161; ngưỡng vượt màn là 0,8.

## 1. Kết quả tốt nhất hiện có

| Mạng | Checkpoint | 100 lượt, seed 7 | 20 lượt, seed 2026 |
|---|---|---|---|
| Chỉ học 1-1 (**bản nộp**, `submission/1-1.pt`) | `runs/1-1/run-1`, bước 1.200.128 | 89/100 | 19/20 |
| Một mạng 2 màn (1-1 và 4-1) | `runs/1-1+4-1/from-scratch`, bước 2.904.064 | 99/100 | 20/20 |
| Một mạng 8 màn | `runs/8-levels/from-scratch-10m`, bước 9.900.032 | 99/100 | 20/20 |
| Một mạng 8 màn (checkpoint cuối) | `runs/8-levels/from-scratch-10m`, bước 9.998.336 | 99/100 | 19/20 |

Màn 1-1 đã vượt ngưỡng ở cả ba mạng. Bản nộp vẫn là mạng chỉ học 1-1.

## 2. Các lượt chạy có huấn luyện 1-1

Mọi lượt đều dùng seed 1, 8 môi trường × 512 bước, γ = 0,9, bộ hành động `simple`.

| Lượt chạy | Số màn | Tổng bước | Bước riêng trên 1-1 | Tốc độ học | Chạm cờ lúc huấn luyện, cuối lượt |
|---|---|---|---|---|---|
| `legacy/runs/trial-100k` (bản tự viết) | 1 | 1.503.232 | 1.503.232 | 1,5e-4 giảm về 1e-4 | 88% |
| `runs/1-1/run-1` | 1 | 1.200.128 | 1.200.128 | 1,5e-4, lịch tính cho 2 triệu bước | 75% |
| `runs/1-1+4-1/from-scratch` | 2 | 3.002.368 | 1.501.184 | 2e-4 giảm về 0 | 99% |
| `runs/multi-level/restart-test` | 8 | 372.736 (nối tiếp mạng 2 màn) | 46.592 thêm | 1,4e-4 | 86% |
| `runs/8-levels/from-scratch-10m` | 8 | 9.998.336 | 955.357 | 2e-4 giảm về 0 | 96% |

- Bản tự viết dùng mã và cách xử lý khung hình khác kho của thầy; checkpoint của nó không dùng được với `evaluate.py` hiện tại.
- Ở lượt 8 màn, 1-1 nhận 1/8 số bước tới bước 2.002.944 (mỗi màn một môi trường); sau đó phần bước chia theo tỉ lệ hỏng lúc huấn luyện (sàn 6%, trần 30%, ghi trong `allocation.csv`) nên 1-1 chỉ nhận 8–10% số bước của mỗi triệu bước, 9% của cả chặng.
- `runs/4-1/transfer` khởi tạo từ checkpoint 1-1 nhưng chỉ học 4-1, nên không tính là lượt huấn luyện 1-1 (xem Mục 6).

## 3. Đường cong học

![Tỉ lệ chạm cờ lúc huấn luyện của màn 1-1 qua các lượt chạy](figures/1-1-training.png)

Hình vẽ bằng `python experiments/level_curves.py 1-1`.

- **Theo tổng số bước, càng nhiều màn thì 1-1 càng chậm đạt ngưỡng.** Hai lượt một màn chạm 80% ở khoảng 0,8–0,9 triệu bước, lượt 2 màn ở 1,2 triệu, lượt 8 màn ở 3,0 triệu.
- **Theo số bước riêng trên 1-1 thì ngược lại.** Lượt 8 màn chạm 80% sau 0,35 triệu bước riêng, lượt 2 màn sau 0,6 triệu, lượt một màn sau 0,8 triệu.
- **Mức cuối cao hơn ở các lượt học nhiều màn và chạy hết lịch tốc độ học.** Lượt 2 màn kết thúc ở 99%, lượt 8 màn ở 96% (50 dòng cuối của `progress.csv` nằm trong khoảng 91–96%), hai lượt một màn dừng quanh 85–88%.
- **Thêm màn mới đột ngột làm 1-1 tụt.** Ở lượt thử nối tiếp, 1-1 đi từ 100% xuống 86% sau 373 nghìn bước.

Giới hạn: mỗi lượt chỉ một seed và các lượt khác nhau nhiều thứ cùng lúc (số màn, số bước, tốc độ học, lịch entropy), nên các nhận xét trên là quan sát, chưa phải kết luận về nguyên nhân.

## 4. Số chấm theo checkpoint

### Lượt một màn (`runs/1-1/run-1`)

| Bước | 500k | 600k | 700k | 800k | 900k | 1,0M | 1,1M | 1,2M |
|---|---|---|---|---|---|---|---|---|
| Chạm cờ /100 (seed 7) | 46 | 71 | 66 | 83 | 81 | 84 | 80 | **89** |
| Chạm cờ /20 (seed 2026) | – | – | – | 17 | 16 | 19 | 19 | **19** |

### Một mạng 2 màn (`runs/1-1+4-1/from-scratch`), cột 1-1

| Bước | 0,5M | 1,0M | 1,2M | 1,5M | 2,0M | 2,5M | 2,9M | 3,0M |
|---|---|---|---|---|---|---|---|---|
| Chạm cờ /100 (seed 7) | 22 | 73 | 92 | 88 | 92 | 97 | **99** | 100 |
| Chạm cờ /20 (seed 2026) | 2 | 15 | 16 | 15 | 18 | 20 | **20** | 18 |

Bảng đủ 30 checkpoint ở Mục 10 của `docs/nhat-ky-trong-tu.md`.

### Lượt thử nối tiếp (`runs/multi-level/restart-test`)

| Bước sau khi thêm 6 màn mới | 0 | 24.576 | 106.496 | 208.896 | 372.736 |
|---|---|---|---|---|---|
| Chạm cờ /100 (seed 7) | 99 | 98 | 88 | 75 | 80 |

### Một mạng 8 màn (`runs/8-levels/from-scratch-10m`)

| Bước | 1,90M | 2,00M | 3,40M | 3,50M | 6,20M | 6,30M | 8,00M | 9,00M | 9,90M | 9,998M (cuối) |
|---|---|---|---|---|---|---|---|---|---|---|
| Bước riêng trên 1-1 | 237.568 | 250.368 | 383.507 | 390.830 | 628.356 | 636.918 | 786.838 | 867.988 | 947.784 | 955.357 |
| Chạm cờ /100 (seed 7) | 13 | 36 | 67 | 79 | 95 | 86 | 83 | 85 | 99 | 99 |
| Chạm cờ /100 (seed 11) | – | – | – | – | – | – | – | – | – | 96 |
| x trung bình (seed 7) | 2099 | 2352 | 2719 | 2855 | 3083 | 2966 | 2923 | 2998 | 3154 | 3154 |
| Chạm cờ /20 (seed 2026) | – | – | – | – | – | – | – | – | 20 | 19 |

- Checkpoint cuối được chấm kiểm tra thêm 100 lượt seed 11: 96/100 (96 chạm cờ, 2 chạm địch, 2 rơi hố). Gộp với seed 7: 195/200 = 97,5%, khoảng tin cậy 95% 94,3–98,9% (công thức Wilson). Tệp ở `runs/8-levels/from-scratch-10m/eval/step-009998336-100ep-seed11/deaths-1-1/`. Video lượt chạm cờ đầu tiên theo lệnh chấm: `runs/8-levels/from-scratch-10m/eval/step-009998336-20ep-seed2026/videos/1-1.mp4`.
- Từ bước 6,30M tới cuối lượt, số chạm cờ /100 tăng từ 86 lên 99.
- Hai checkpoint cách nhau khoảng 100 nghìn bước có thể chênh 23 điểm (bước 1,90M và 2,00M: 13 và 36), nên không đọc xu hướng từ một checkpoint đơn lẻ.

## 5. Mario hỏng ở đâu

### Bản nộp (bước 1.200.128, 100 lượt seed 7, 11 lượt hỏng)

| Vị trí x | Số lượt | Chuyện xảy ra |
|---|---|---|
| 1791–1793 | 3 | hai Goomba ngay sau con Koopa ở x≈1700; Mario đáp xuống ngay trước mặt chúng |
| 1978–2025 | 3 | nhóm Goomba dưới dãy gạch; rơi cạnh một con hoặc nảy từ con này sang con khác |
| 2468–2472 | 3 | hố giữa hai cầu thang; không bật nhảy hoặc nhảy thiếu |
| 2743–2759 | 2 | hai Goomba giữa hai ống cuối, dưới dãy gạch thấp; cú nhảy bị gạch chặn |

- Chọn hành động xác suất cao nhất từ cùng các điểm xuất phát thì đạt 97/100, nên phần lớn thất bại là nhiễu do lấy mẫu ở chỗ cần chuỗi hành động chính xác.
- Ảnh từng trường hợp ở `runs/1-1/run-1/eval/failure-analysis-001200128/`.

### Mạng 8 màn (100 lượt seed 7 mỗi checkpoint)

| Bước | Chạm cờ | Chết vì địch | Rơi hố |
|---|---|---|---|
| 2,00M | 36 | 60 | 4 |
| 3,40M | 67 | 26 | 7 |
| 3,50M | 79 | 13 | 8 |
| 6,20M | 95 | 4 | 1 |
| 6,30M | 86 | 11 | 3 |
| 8,00M | 83 | 13 | 4 |
| 9,00M | 85 | 13 | 2 |
| 9,90M | 99 | 0 | 1 |
| 9,998M | 99 | 0 | 1 |

Số lượt chết vì địch giảm mạnh nhất. Ở checkpoint cuối (bước 9.998.336) chỉ còn 1 lượt hỏng: rơi hố ở x = 2471, đúng hố giữa hai cầu thang mà bản nộp cũng hỏng 3 lượt (x 2468–2472). Chưa xem khung hình các lượt hỏng của mạng 8 màn ở 1-1.

## 6. Mạng không học 1-1 thì chơi 1-1 thế nào

| Mạng | Chạm cờ ở 1-1 (20 lượt, seed 2026) |
|---|---|
| Chỉ học 4-1 (`runs/4-1/from-scratch`, bước 1.200.128) | 0/20 |
| Chỉ học 2-3 (`runs/2-3/from-scratch-3m`, bước 2.801.664) | 0/20 |
| Chuyển giao 1-1 → 4-1, sau 100 nghìn bước chỉ học 4-1 | 10/20 |
| Chuyển giao 1-1 → 4-1, sau 200 nghìn bước chỉ học 4-1 | 2/20 |
| Chuyển giao 1-1 → 4-1, từ 300 nghìn bước trở đi (tới 1,3 triệu) | 0–2/20 |

Kỹ năng ở màn khác không tự mang sang 1-1, và học lần lượt làm quên 1-1 sau 100–200 nghìn bước.

## 7. Điều chưa biết

- Vì sao tính theo bước riêng thì mạng nhiều màn học 1-1 nhanh hơn: do học chung, do tốc độ học cao hơn (2e-4 so với 1,5e-4), hay do lịch entropy. Chưa có thí nghiệm tách riêng.
- Mọi kết quả mới có một seed.

## 8. Nhật ký cập nhật

| Ngày | Thay đổi |
|---|---|
| 24/11/2026 | Tạo báo cáo từ năm lượt chạy đã có. Lượt 8 màn đang ở bước 9,91 triệu; số chấm của nó mới tới mốc 6,3 triệu. |
| 24/11/2026 | Điền số của lượt 8 màn sau khi lượt kết thúc: 100 lượt seed 7 ở các bước 8,00M, 9,00M, 9,90M, 9,998M; lệnh chấm ở các bước 9,90M, 9,998M; vẽ lại hình. Sửa hai số cũ: phần bước của 1-1 ở chặng chia theo tỉ lệ hỏng (trước ghi khoảng 7–9%) và bước riêng của lượt thử nối tiếp (46.592, trước ghi khoảng 46.600). |
| 24/11/2026 | Thêm lần chấm kiểm tra 100 lượt seed 11 ở checkpoint cuối của lượt 8 màn (96/100, gộp hai seed 195/200) và đường dẫn video quay bằng lệnh chấm. |
