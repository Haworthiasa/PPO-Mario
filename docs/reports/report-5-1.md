# Báo cáo màn 5-1

Cập nhật lần cuối: 24/11/2026, 10:30. Lượt 8 màn đã kết thúc ở bước 9.998.336; 10 checkpoint của nó đã được chấm 100 lượt seed 7 trên 5-1, 2 checkpoint đã được chấm theo lệnh của đề. Checkpoint cuối đã được chấm thêm 100 lượt seed 11.

Quy ước: **bước** là bước tác tử. **Số chấm** là kết quả của `evaluate.py` (100 lượt seed 7 để ra quyết định, 20 lượt seed 2026 là đúng lệnh chấm của đề). **Số lúc huấn luyện** là tỉ lệ chạm cờ của 100 lượt huấn luyện gần nhất trong `progress.csv`; hai loại số có thể lệch nhau hơn 10 điểm. Cờ của 5-1 ở x = 3177; ngưỡng vượt màn là 0,8.

## 1. Kết quả tốt nhất hiện có

| Mạng | Checkpoint | 100 lượt, seed 7 | 20 lượt, seed 2026 |
|---|---|---|---|
| Một mạng 8 màn | `runs/8-levels/from-scratch-10m`, bước 9.900.032 | 84/100 | 17/20 |
| Một mạng 8 màn (checkpoint cuối) | `runs/8-levels/from-scratch-10m`, bước 9.998.336 | 91/100 | 16/20 |

Ở checkpoint cuối (bước 9.998.336), mạng 8 màn đạt 91/100 theo 100 lượt seed 7 và 16/20 theo lệnh chấm. Màn 5-1 vượt ngưỡng 0,8 ở mạng này. Màn 5-1 không có checkpoint nộp; số ở đây chỉ dùng cho báo cáo.

## 2. Các lượt chạy có huấn luyện 5-1

Mọi lượt đều dùng seed 1, 8 môi trường × 512 bước, γ = 0,9, bộ hành động `simple`.

| Lượt chạy | Số màn | Tổng bước | Bước riêng trên 5-1 | Tốc độ học | Chạm cờ lúc huấn luyện, cuối lượt |
|---|---|---|---|---|---|
| `runs/multi-level/restart-test` | 8 | 372.736 (nối tiếp mạng 2 màn) | 46.592 | 1,4e-4 | 0% |
| `runs/8-levels/from-scratch-10m` | 8 | 9.998.336 | 1.311.858 | 2e-4 giảm về 0 | 88% |

- Ở lượt 8 màn, mỗi màn chạy trên một môi trường tới bước 2.002.944, nên 5-1 nhận 1/8 số bước đó. Sau đó phần bước chia theo tỉ lệ hỏng lúc huấn luyện (sàn 6%, trần 30%, ghi trong `allocation.csv`): 5-1 nhận 13% số bước của chặng này (14% ở triệu bước thứ 3, 14% ở triệu bước thứ 7, 12% ở triệu bước thứ 10).
- Bước riêng trên 5-1 = 1/8 của 2.002.944 bước đầu cộng dồn cột `steps_5-1` của `allocation.csv`.
- Lượt thử nối tiếp lấy mạng 2 màn (1-1 và 4-1) ở bước 2.904.064 rồi cho học 8 màn 1-1, 4-1, 2-1, 3-1, 3-2, 5-1, 6-1, 8-1, mỗi màn một môi trường, tốc độ học 1,4e-4. Nó bị dừng tay sau 372.736 bước; chỉ còn `progress.csv` và `summary.txt`, checkpoint đã xóa.

## 3. Đường cong học

![Tỉ lệ chạm cờ lúc huấn luyện của màn 5-1 qua các lượt chạy](figures/5-1-training.png)

Hình vẽ bằng `python experiments/level_curves.py 5-1`.

- **Màn 5-1 đi lên chậm và đều.** Số lúc huấn luyện lần đầu khác 0 ở 2,15 triệu bước (0,27 triệu bước riêng), chạm 50% ở 5,66 triệu (0,76 triệu riêng), chạm 80% lần đầu ở 8,08 triệu (1,08 triệu riêng), ở yên trên 80% từ 9,27 triệu (1,23 triệu riêng).
- Cao nhất 93% ở 9,67 triệu bước, cuối lượt 88%; 50 dòng cuối nằm trong khoảng 84–90%. Đường cong vẫn đang lên khi tốc độ học về 0.
- Lượt thử nối tiếp (mạng đã học 1-1 và 4-1) và lượt 8 màn ở cùng 46.592 bước riêng trên 5-1 (bước 372.736 của lượt 8 màn): x trung bình lúc huấn luyện 919 so với 635, cả hai chưa chạm cờ.

Giới hạn: mỗi lượt chỉ một seed, và trong lượt 8 màn phần bước của từng màn đổi theo thời gian cùng lúc với tốc độ học và hệ số entropy. Các nhận xét trên là quan sát, chưa phải kết luận về nguyên nhân.

## 4. Số chấm theo checkpoint

### Một mạng 8 màn (`runs/8-levels/from-scratch-10m`)

| Bước | 1,90M | 2,00M | 3,40M | 3,50M | 6,20M | 6,30M | 8,00M | 9,00M | 9,90M | 9,998M (cuối) |
|---|---|---|---|---|---|---|---|---|---|---|
| Bước riêng trên 5-1 | 237.568 | 250.368 | 440.852 | 456.458 | 829.574 | 840.644 | 1.075.487 | 1.195.864 | 1.301.247 | 1.311.858 |
| Chạm cờ /100 (seed 7) | 0 | 1 | 18 | 19 | 63 | 64 | 74 | 87 | 84 | 91 |
| Chạm cờ /100 (seed 11) | – | – | – | – | – | – | – | – | – | 88 |
| x trung bình (seed 7) | 1476 | 1580 | 2152 | 2241 | 2468 | 2689 | 2877 | 2916 | 2922 | 3051 |
| Chạm cờ /20 (seed 2026) | – | – | – | – | – | – | – | – | 17 | 16 |

- Checkpoint cuối được chấm kiểm tra thêm 100 lượt seed 11: 88/100 (88 chạm cờ, 12 chạm địch). Gộp với seed 7: 179/200 = 89,5%, khoảng tin cậy 95% 84,5–93,0% (công thức Wilson). Tệp ở `runs/8-levels/from-scratch-10m/eval/step-009998336-100ep-seed11/deaths-5-1/`. Video lượt chạm cờ đầu tiên theo lệnh chấm: `runs/8-levels/from-scratch-10m/eval/step-009998336-20ep-seed2026/videos/5-1.mp4`.
- Từ bước 6,30M tới cuối lượt, số chạm cờ /100 tăng từ 64 lên 91.
- Hai checkpoint cách nhau khoảng 100 nghìn bước có thể chênh 7 điểm (bước 9,90M và 9,998M: 84 và 91), nên không đọc xu hướng từ một checkpoint đơn lẻ.

### Lượt thử nối tiếp (`runs/multi-level/restart-test`)

Chưa chấm trên 5-1, và không chấm lại được vì checkpoint đã xóa. Số lúc huấn luyện ở cuối lượt (sau 46.592 bước riêng): chạm cờ 0%, x trung bình 919.

## 5. Mario hỏng ở đâu

### Lý do kết thúc lượt theo checkpoint (100 lượt seed 7)

| Bước | Chạm cờ | Chết vì địch | Rơi hố |
|---|---|---|---|
| 2,00M | 1 | 57 | 42 |
| 3,40M | 18 | 49 | 33 |
| 3,50M | 19 | 62 | 19 |
| 6,20M | 63 | 29 | 8 |
| 6,30M | 64 | 31 | 5 |
| 8,00M | 74 | 20 | 6 |
| 9,00M | 87 | 12 | 1 |
| 9,90M | 84 | 16 | 0 |
| 9,998M | 91 | 8 | 1 |

Lý do đọc từ trạng thái trình giả lập bằng `experiments/death_look.py`, cùng các lượt với `evaluate.py`. "Chết vì địch" là chết khi còn trong màn hình (địch, cây ăn thịt, đạn); "hết giờ" là đồng hồ của trò chơi về 0, thường do Mario kẹt trước một vật cản. Checkpoint 1,90M chỉ có tệp của `evaluate.py`, không có lý do.

### Chỗ kết thúc ở checkpoint cuối (bước 9.998.336, 9 lượt hỏng trên 100)

| Khoảng x | Số lượt | Lý do kết thúc | x cuối cụ thể |
|---|---|---|---|
| 600–699 | 1 | 1 chạm địch | 645 |
| 700–799 | 2 | 2 chạm địch | 703 |
| 1800–1899 | 1 | 1 rơi hố | 1845 |
| 2100–2199 | 2 | 2 chạm địch | 2129–2132 |
| 2500–2599 | 1 | 1 chạm địch | 2516 |
| 2600–2699 | 2 | 2 chạm địch | 2613–2673 |

Bảng đủ ở `runs/8-levels/from-scratch-10m/eval/step-009998336-100ep-seed7/deaths-5-1/look.txt`.

- Ở bước 6,30M các lượt hỏng rải khắp màn; hai chỗ đông nhất là x 702–704 (6 lượt chạm địch) và x 2805–2881 (7 lượt chạm địch).
- Checkpoint cuối còn 9 lượt hỏng (8 chạm địch, 1 rơi hố), không chỗ nào quá 2 lượt. Ở x 2805–2881 không còn lượt nào; ở x = 703 còn 2.
- Chưa xem khung hình các lượt hỏng của mạng 8 màn ở 5-1.

## 6. Mạng không học 5-1 thì chơi 5-1 thế nào

| Mạng | Chạm cờ ở 5-1 (20 lượt, seed 2026) | x trung bình | x xa nhất |
|---|---|---|---|
| Chỉ học 1-1 (`runs/1-1/run-1`, bước 1.200.128) | 0/20 | 403 | 940 |
| Chỉ học 4-1 (`runs/4-1/from-scratch`, bước 1.200.128) | 0/20 | 493 | 807 |
| Chỉ học 2-3 (`runs/2-3/from-scratch-3m`, bước 2.801.664) | 0/20 | 229 | 705 |
| Một mạng 2 màn 1-1 và 4-1, bước 1.503.232 | 0/20 | 352 | 1762 |
| Một mạng 2 màn 1-1 và 4-1, bước 1.900.544 | 0/20 | 378 | 812 |
| Một mạng 2 màn 1-1 và 4-1, bước 2.904.064 | 0/20 | 443 | 716 |

Không mạng nào chạm cờ ở 5-1 khi chưa học màn này; x xa nhất là 1762 (một lượt của mạng 2 màn ở bước 1.503.232).

## 7. Điều chưa biết

- Đường cong lúc huấn luyện vẫn đi lên khi tốc độ học về 0. Ở 2-3, dấu hiệu này đi cùng việc thiếu bước (điều 9, Mục 14 của `docs/nhat-ky-trong-tu.md`); ở 5-1 chưa thử chạy thêm.
- Một mạng chỉ học 5-1 cần bao nhiêu bước để vượt ngưỡng: chưa có lượt một màn nào cho 5-1, nên chưa so được học chung với học riêng như ở 1-1 và 4-1.
- Mọi kết quả mới có một seed huấn luyện.

Các hướng mở rộng chung cho mọi màn ở `docs/extension-directions.md`.

## 8. Nhật ký cập nhật

| Ngày | Thay đổi |
|---|---|
| 24/11/2026 | Tạo báo cáo từ lượt 8 màn (đã kết thúc) và lượt thử nối tiếp. |
| 24/11/2026 | Thêm lần chấm kiểm tra 100 lượt seed 11 ở checkpoint cuối của lượt 8 màn (88/100, gộp hai seed 179/200) và đường dẫn video quay bằng lệnh chấm. |
