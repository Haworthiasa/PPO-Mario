# Báo cáo màn 3-2

Cập nhật lần cuối: 24/11/2026, 10:30. Lượt 8 màn đã kết thúc ở bước 9.998.336; 10 checkpoint của nó đã được chấm 100 lượt seed 7 trên 3-2, 2 checkpoint đã được chấm theo lệnh của đề. Checkpoint cuối đã được chấm thêm 100 lượt seed 11.

Quy ước: **bước** là bước tác tử. **Số chấm** là kết quả của `evaluate.py` (100 lượt seed 7 để ra quyết định, 20 lượt seed 2026 là đúng lệnh chấm của đề). **Số lúc huấn luyện** là tỉ lệ chạm cờ của 100 lượt huấn luyện gần nhất trong `progress.csv`; hai loại số có thể lệch nhau hơn 10 điểm. Cờ của 3-2 ở x = 3337; ngưỡng vượt màn là 0,8.

## 1. Kết quả tốt nhất hiện có

| Mạng | Checkpoint | 100 lượt, seed 7 | 20 lượt, seed 2026 |
|---|---|---|---|
| Một mạng 8 màn (cao nhất theo 100 lượt) | `runs/8-levels/from-scratch-10m`, bước 9.900.032 | 98/100 | 20/20 |
| Một mạng 8 màn (checkpoint cuối) | `runs/8-levels/from-scratch-10m`, bước 9.998.336 | 96/100 | 20/20 |

Ở checkpoint cuối (bước 9.998.336), mạng 8 màn đạt 96/100 theo 100 lượt seed 7 và 20/20 theo lệnh chấm. Màn 3-2 vượt ngưỡng 0,8 ở mạng này. Checkpoint cao nhất theo 100 lượt là bước 9.900.032 (98/100). Màn 3-2 không có checkpoint nộp; số ở đây chỉ dùng cho báo cáo.

## 2. Các lượt chạy có huấn luyện 3-2

Mọi lượt đều dùng seed 1, 8 môi trường × 512 bước, γ = 0,9, bộ hành động `simple`.

| Lượt chạy | Số màn | Tổng bước | Bước riêng trên 3-2 | Tốc độ học | Chạm cờ lúc huấn luyện, cuối lượt |
|---|---|---|---|---|---|
| `runs/multi-level/restart-test` | 8 | 372.736 (nối tiếp mạng 2 màn) | 46.592 | 1,4e-4 | 1% |
| `runs/8-levels/from-scratch-10m` | 8 | 9.998.336 | 999.830 | 2e-4 giảm về 0 | 99% |

- Ở lượt 8 màn, mỗi màn chạy trên một môi trường tới bước 2.002.944, nên 3-2 nhận 1/8 số bước đó. Sau đó phần bước chia theo tỉ lệ hỏng lúc huấn luyện (sàn 6%, trần 30%, ghi trong `allocation.csv`): 3-2 nhận 9% số bước của chặng này (13% ở triệu bước thứ 3, 8% ở triệu bước thứ 7, 8% ở triệu bước thứ 10).
- Bước riêng trên 3-2 = 1/8 của 2.002.944 bước đầu cộng dồn cột `steps_3-2` của `allocation.csv`.
- Lượt thử nối tiếp lấy mạng 2 màn (1-1 và 4-1) ở bước 2.904.064 rồi cho học 8 màn 1-1, 4-1, 2-1, 3-1, 3-2, 5-1, 6-1, 8-1, mỗi màn một môi trường, tốc độ học 1,4e-4. Nó bị dừng tay sau 372.736 bước; chỉ còn `progress.csv` và `summary.txt`, checkpoint đã xóa.

## 3. Đường cong học

![Tỉ lệ chạm cờ lúc huấn luyện của màn 3-2 qua các lượt chạy](figures/3-2-training.png)

Hình vẽ bằng `python experiments/level_curves.py 3-2`.

- **Màn 3-2 kết thúc với số lúc huấn luyện cao nhất trong tám màn (99%).** Số lúc huấn luyện lần đầu khác 0 ở 0,55 triệu bước (0,07 triệu bước riêng), chạm 50% ở 3,06 triệu (0,39 triệu riêng), chạm 80% lần đầu ở 4,37 triệu (0,52 triệu riêng), ở yên trên 80% từ 7,50 triệu (0,79 triệu riêng). Cuối lượt 99%; 50 dòng cuối nằm trong khoảng 96–99%.
- **Một lần tụt mạnh quanh mốc 2 triệu bước:** từ 29% xuống 3%. Số chấm cũng tụt cùng lúc: 28/100 ở bước 1,90M, 3/100 ở bước 2,00M (Mục 4). Checkpoint 2,00M có trước lúc đổi cách chia bước, nên lần tụt này không do việc đổi đó.
- Lượt thử nối tiếp (mạng đã học 1-1 và 4-1) và lượt 8 màn ở cùng 46.592 bước riêng trên 3-2 (bước 372.736 của lượt 8 màn): x trung bình lúc huấn luyện 2214 so với 974, chạm cờ 1% so với 0%.

Giới hạn: mỗi lượt chỉ một seed, và trong lượt 8 màn phần bước của từng màn đổi theo thời gian cùng lúc với tốc độ học và hệ số entropy. Các nhận xét trên là quan sát, chưa phải kết luận về nguyên nhân.

## 4. Số chấm theo checkpoint

### Một mạng 8 màn (`runs/8-levels/from-scratch-10m`)

| Bước | 1,90M | 2,00M | 3,40M | 3,50M | 6,20M | 6,30M | 8,00M | 9,00M | 9,90M | 9,998M (cuối) |
|---|---|---|---|---|---|---|---|---|---|---|
| Bước riêng trên 3-2 | 237.568 | 250.368 | 421.392 | 433.436 | 680.015 | 689.224 | 838.202 | 920.100 | 990.724 | 999.830 |
| Chạm cờ /100 (seed 7) | 28 | 3 | 62 | 56 | 61 | 75 | 85 | 94 | 98 | 96 |
| Chạm cờ /100 (seed 11) | – | – | – | – | – | – | – | – | – | 98 |
| x trung bình (seed 7) | 2222 | 1808 | 2878 | 2833 | 2683 | 3000 | 3072 | 3264 | 3310 | 3250 |
| Chạm cờ /20 (seed 2026) | – | – | – | – | – | – | – | – | 20 | 20 |

- Checkpoint cuối được chấm kiểm tra thêm 100 lượt seed 11: 98/100 (98 chạm cờ, 2 chạm địch). Gộp với seed 7: 194/200 = 97,0%, khoảng tin cậy 95% 93,6–98,6% (công thức Wilson). Tệp ở `runs/8-levels/from-scratch-10m/eval/step-009998336-100ep-seed11/deaths-3-2/`. Video lượt chạm cờ đầu tiên theo lệnh chấm: `runs/8-levels/from-scratch-10m/eval/step-009998336-20ep-seed2026/videos/3-2.mp4`.
- Từ bước 6,30M tới cuối lượt, số chạm cờ /100 tăng từ 75 lên 96.
- Hai checkpoint cách nhau khoảng 100 nghìn bước có thể chênh 25 điểm (bước 1,90M và 2,00M: 28 và 3), nên không đọc xu hướng từ một checkpoint đơn lẻ.

### Lượt thử nối tiếp (`runs/multi-level/restart-test`)

Chưa chấm trên 3-2, và không chấm lại được vì checkpoint đã xóa. Số lúc huấn luyện ở cuối lượt (sau 46.592 bước riêng): chạm cờ 1%, x trung bình 2214.

## 5. Mario hỏng ở đâu

### Lý do kết thúc lượt theo checkpoint (100 lượt seed 7)

| Bước | Chạm cờ | Chết vì địch | Rơi hố |
|---|---|---|---|
| 2,00M | 3 | 29 | 68 |
| 3,40M | 62 | 21 | 17 |
| 3,50M | 56 | 30 | 14 |
| 6,20M | 61 | 30 | 9 |
| 6,30M | 75 | 15 | 10 |
| 8,00M | 85 | 5 | 10 |
| 9,00M | 94 | 1 | 5 |
| 9,90M | 98 | 2 | 0 |
| 9,998M | 96 | 3 | 1 |

Lý do đọc từ trạng thái trình giả lập bằng `experiments/death_look.py`, cùng các lượt với `evaluate.py`. "Chết vì địch" là chết khi còn trong màn hình (địch, cây ăn thịt, đạn); "hết giờ" là đồng hồ của trò chơi về 0, thường do Mario kẹt trước một vật cản. Checkpoint 1,90M chỉ có tệp của `evaluate.py`, không có lý do.

### Chỗ kết thúc ở checkpoint cuối (bước 9.998.336, 4 lượt hỏng trên 100)

| Khoảng x | Số lượt | Lý do kết thúc | x cuối cụ thể |
|---|---|---|---|
| 400–499 | 1 | 1 chạm địch | 496 |
| 1100–1199 | 2 | 2 chạm địch | 1105 |
| 1900–1999 | 1 | 1 rơi hố | 1988 |

Bảng đủ ở `runs/8-levels/from-scratch-10m/eval/step-009998336-100ep-seed7/deaths-3-2/look.txt`.

- Ở bước 6,30M, chỗ đông nhất là x 2012–2077 với 10 lượt rơi hố. Ở checkpoint cuối không còn lượt nào hỏng ở đó.
- Checkpoint cuối còn 4 lượt hỏng: x = 496 (chạm địch), x = 1105 (2 lượt chạm địch), x = 1988 (rơi hố).
- Chưa xem khung hình các lượt hỏng của mạng 8 màn ở 3-2.

## 6. Mạng không học 3-2 thì chơi 3-2 thế nào

| Mạng | Chạm cờ ở 3-2 (20 lượt, seed 2026) | x trung bình | x xa nhất |
|---|---|---|---|
| Chỉ học 1-1 (`runs/1-1/run-1`, bước 1.200.128) | 0/20 | 425 | 1107 |
| Chỉ học 4-1 (`runs/4-1/from-scratch`, bước 1.200.128) | 0/20 | 264 | 660 |
| Chỉ học 2-3 (`runs/2-3/from-scratch-3m`, bước 2.801.664) | 0/20 | 341 | 1104 |
| Một mạng 2 màn 1-1 và 4-1, bước 1.503.232 | 0/20 | 694 | 1876 |
| Một mạng 2 màn 1-1 và 4-1, bước 1.900.544 | 0/20 | 772 | 2082 |
| Một mạng 2 màn 1-1 và 4-1, bước 2.904.064 | 0/20 | 840 | 1312 |

Không mạng nào chạm cờ ở 3-2 khi chưa học màn này. Mạng 2 màn 1-1 và 4-1 đi xa nhất (x trung bình 694–840, xa nhất 2082), xa hơn cả ba mạng một màn.

## 7. Điều chưa biết

- Vì sao số chấm và số lúc huấn luyện của 3-2 cùng tụt mạnh quanh mốc 2 triệu bước.
- Một mạng chỉ học 3-2 cần bao nhiêu bước để vượt ngưỡng: chưa có lượt một màn nào cho 3-2, nên chưa so được học chung với học riêng như ở 1-1 và 4-1.
- Mọi kết quả mới có một seed huấn luyện.

Các hướng mở rộng chung cho mọi màn ở `docs/extension-directions.md`.

## 8. Nhật ký cập nhật

| Ngày | Thay đổi |
|---|---|
| 24/11/2026 | Tạo báo cáo từ lượt 8 màn (đã kết thúc) và lượt thử nối tiếp. |
| 24/11/2026 | Thêm lần chấm kiểm tra 100 lượt seed 11 ở checkpoint cuối của lượt 8 màn (98/100, gộp hai seed 194/200) và đường dẫn video quay bằng lệnh chấm. |
