# Báo cáo màn 3-1

Cập nhật lần cuối: 24/11/2026, 10:30. Lượt 8 màn đã kết thúc ở bước 9.998.336; 10 checkpoint của nó đã được chấm 100 lượt seed 7 trên 3-1, 2 checkpoint đã được chấm theo lệnh của đề. Checkpoint cuối đã được chấm thêm 100 lượt seed 11.

Quy ước: **bước** là bước tác tử. **Số chấm** là kết quả của `evaluate.py` (100 lượt seed 7 để ra quyết định, 20 lượt seed 2026 là đúng lệnh chấm của đề). **Số lúc huấn luyện** là tỉ lệ chạm cờ của 100 lượt huấn luyện gần nhất trong `progress.csv`; hai loại số có thể lệch nhau hơn 10 điểm. Cờ của 3-1 ở x = 3193; ngưỡng vượt màn là 0,8.

## 1. Kết quả tốt nhất hiện có

| Mạng | Checkpoint | 100 lượt, seed 7 | 20 lượt, seed 2026 |
|---|---|---|---|
| Một mạng 8 màn | `runs/8-levels/from-scratch-10m`, bước 9.900.032 | 97/100 | 20/20 |
| Một mạng 8 màn (checkpoint cuối) | `runs/8-levels/from-scratch-10m`, bước 9.998.336 | 97/100 | 20/20 |

Ở checkpoint cuối (bước 9.998.336), mạng 8 màn đạt 97/100 theo 100 lượt seed 7 và 20/20 theo lệnh chấm. Màn 3-1 vượt ngưỡng 0,8 ở mạng này. Màn 3-1 không có checkpoint nộp; số ở đây chỉ dùng cho báo cáo.

## 2. Các lượt chạy có huấn luyện 3-1

Mọi lượt đều dùng seed 1, 8 môi trường × 512 bước, γ = 0,9, bộ hành động `simple`.

| Lượt chạy | Số màn | Tổng bước | Bước riêng trên 3-1 | Tốc độ học | Chạm cờ lúc huấn luyện, cuối lượt |
|---|---|---|---|---|---|
| `runs/multi-level/restart-test` | 8 | 372.736 (nối tiếp mạng 2 màn) | 46.592 | 1,4e-4 | 0% |
| `runs/8-levels/from-scratch-10m` | 8 | 9.998.336 | 1.244.155 | 2e-4 giảm về 0 | 93% |

- Ở lượt 8 màn, mỗi màn chạy trên một môi trường tới bước 2.002.944, nên 3-1 nhận 1/8 số bước đó. Sau đó phần bước chia theo tỉ lệ hỏng lúc huấn luyện (sàn 6%, trần 30%, ghi trong `allocation.csv`): 3-1 nhận 12% số bước của chặng này (14% ở triệu bước thứ 3, 11% ở triệu bước thứ 7, 10% ở triệu bước thứ 10).
- Bước riêng trên 3-1 = 1/8 của 2.002.944 bước đầu cộng dồn cột `steps_3-1` của `allocation.csv`.
- Lượt thử nối tiếp lấy mạng 2 màn (1-1 và 4-1) ở bước 2.904.064 rồi cho học 8 màn 1-1, 4-1, 2-1, 3-1, 3-2, 5-1, 6-1, 8-1, mỗi màn một môi trường, tốc độ học 1,4e-4. Nó bị dừng tay sau 372.736 bước; chỉ còn `progress.csv` và `summary.txt`, checkpoint đã xóa.

## 3. Đường cong học

![Tỉ lệ chạm cờ lúc huấn luyện của màn 3-1 qua các lượt chạy](figures/3-1-training.png)

Hình vẽ bằng `python experiments/level_curves.py 3-1`.

- **Màn 3-1 bắt đầu muộn rồi đi lên đều.** Số lúc huấn luyện lần đầu khác 0 ở 2,29 triệu bước (0,29 triệu bước riêng), chạm 50% ở 5,30 triệu (0,74 triệu riêng), chạm 80% lần đầu ở 6,16 triệu (0,84 triệu riêng).
- **Chỉ ở yên trên 80% từ 8,86 triệu bước** (1,13 triệu bước riêng). Giữa 6 và 8 triệu bước nó dao động trong khoảng 61–80%. Cuối lượt 93%; 50 dòng cuối nằm trong khoảng 92–94%.
- Lượt thử nối tiếp (mạng đã học 1-1 và 4-1) và lượt 8 màn ở cùng 46.592 bước riêng trên 3-1 (bước 372.736 của lượt 8 màn): x trung bình lúc huấn luyện 681 so với 685, cả hai chưa chạm cờ.

Giới hạn: mỗi lượt chỉ một seed, và trong lượt 8 màn phần bước của từng màn đổi theo thời gian cùng lúc với tốc độ học và hệ số entropy. Các nhận xét trên là quan sát, chưa phải kết luận về nguyên nhân.

## 4. Số chấm theo checkpoint

### Một mạng 8 màn (`runs/8-levels/from-scratch-10m`)

| Bước | 1,90M | 2,00M | 3,40M | 3,50M | 6,20M | 6,30M | 8,00M | 9,00M | 9,90M | 9,998M (cuối) |
|---|---|---|---|---|---|---|---|---|---|---|
| Bước riêng trên 3-1 | 237.568 | 250.368 | 459.919 | 474.770 | 840.890 | 853.806 | 1.045.130 | 1.146.843 | 1.237.343 | 1.244.155 |
| Chạm cờ /100 (seed 7) | 0 | 0 | 1 | 11 | 52 | 66 | 84 | 84 | 97 | 97 |
| Chạm cờ /100 (seed 11) | – | – | – | – | – | – | – | – | – | 100 |
| x trung bình (seed 7) | 1145 | 1383 | 1826 | 1936 | 2711 | 2768 | 2981 | 3055 | 3173 | 3174 |
| Chạm cờ /20 (seed 2026) | – | – | – | – | – | – | – | – | 20 | 20 |

- Checkpoint cuối được chấm kiểm tra thêm 100 lượt seed 11: 100/100 (100 chạm cờ). Gộp với seed 7: 197/200 = 98,5%, khoảng tin cậy 95% 95,7–99,5% (công thức Wilson). Tệp ở `runs/8-levels/from-scratch-10m/eval/step-009998336-100ep-seed11/deaths-3-1/`. Video lượt chạm cờ đầu tiên theo lệnh chấm: `runs/8-levels/from-scratch-10m/eval/step-009998336-20ep-seed2026/videos/3-1.mp4`.
- Từ bước 6,30M tới cuối lượt, số chạm cờ /100 tăng từ 66 lên 97.
- Hai checkpoint cách nhau khoảng 100 nghìn bước có thể chênh 14 điểm (bước 6,20M và 6,30M: 52 và 66), nên không đọc xu hướng từ một checkpoint đơn lẻ.

### Lượt thử nối tiếp (`runs/multi-level/restart-test`)

Chưa chấm trên 3-1, và không chấm lại được vì checkpoint đã xóa. Số lúc huấn luyện ở cuối lượt (sau 46.592 bước riêng): chạm cờ 0%, x trung bình 681.

## 5. Mario hỏng ở đâu

### Lý do kết thúc lượt theo checkpoint (100 lượt seed 7)

| Bước | Chạm cờ | Chết vì địch | Rơi hố |
|---|---|---|---|
| 2,00M | 0 | 74 | 26 |
| 3,40M | 1 | 89 | 10 |
| 3,50M | 11 | 81 | 8 |
| 6,20M | 52 | 40 | 8 |
| 6,30M | 66 | 27 | 7 |
| 8,00M | 84 | 16 | 0 |
| 9,00M | 84 | 16 | 0 |
| 9,90M | 97 | 2 | 1 |
| 9,998M | 97 | 3 | 0 |

Lý do đọc từ trạng thái trình giả lập bằng `experiments/death_look.py`, cùng các lượt với `evaluate.py`. "Chết vì địch" là chết khi còn trong màn hình (địch, cây ăn thịt, đạn); "hết giờ" là đồng hồ của trò chơi về 0, thường do Mario kẹt trước một vật cản. Checkpoint 1,90M chỉ có tệp của `evaluate.py`, không có lý do.

### Chỗ kết thúc ở checkpoint cuối (bước 9.998.336, 3 lượt hỏng trên 100)

| Khoảng x | Số lượt | Lý do kết thúc | x cuối cụ thể |
|---|---|---|---|
| 2300–2399 | 2 | 2 chạm địch | 2338–2339 |
| 2900–2999 | 1 | 1 chạm địch | 2994 |

Bảng đủ ở `runs/8-levels/from-scratch-10m/eval/step-009998336-100ep-seed7/deaths-3-1/look.txt`.

- Ở checkpoint cuối còn 3 lượt hỏng, đều chạm địch: 2 lượt ở x = 2338–2339 và 1 lượt ở x = 2994. Hai chỗ này cũng có lượt hỏng ở bước 6,30M (7 lượt ở x 2338–2339, 3 lượt ở x 2960–2994).
- Ở bước 6,30M còn ba chỗ đông khác: x = 781 (6 lượt rơi hố), x 1070–1081 (3 lượt chạm địch), x 2641–2675 (7 lượt chạm địch). Ở checkpoint cuối không còn lượt nào hỏng ở ba chỗ đó.
- Chưa xem khung hình các lượt hỏng của mạng 8 màn ở 3-1.

## 6. Mạng không học 3-1 thì chơi 3-1 thế nào

| Mạng | Chạm cờ ở 3-1 (20 lượt, seed 2026) | x trung bình | x xa nhất |
|---|---|---|---|
| Chỉ học 1-1 (`runs/1-1/run-1`, bước 1.200.128) | 0/20 | 536 | 962 |
| Chỉ học 4-1 (`runs/4-1/from-scratch`, bước 1.200.128) | chưa chấm | – | – |
| Chỉ học 2-3 (`runs/2-3/from-scratch-3m`, bước 2.801.664) | 0/20 | 460 | 594 |
| Một mạng 2 màn 1-1 và 4-1, bước 2.904.064 | 0/20 | 491 | 608 |

Không mạng nào chạm cờ ở 3-1 khi chưa học màn này; x xa nhất là 962.

## 7. Điều chưa biết

- Vì sao 3-1 phải tới 2,29 triệu bước (0,29 triệu bước riêng) mới chạm cờ lần đầu, muộn hơn 3-2 và 6-1.
- Một mạng chỉ học 3-1 cần bao nhiêu bước để vượt ngưỡng: chưa có lượt một màn nào cho 3-1, nên chưa so được học chung với học riêng như ở 1-1 và 4-1.
- Mọi kết quả mới có một seed huấn luyện.

Các hướng mở rộng chung cho mọi màn ở `docs/extension-directions.md`.

## 8. Nhật ký cập nhật

| Ngày | Thay đổi |
|---|---|
| 24/11/2026 | Tạo báo cáo từ lượt 8 màn (đã kết thúc) và lượt thử nối tiếp. |
| 24/11/2026 | Thêm lần chấm kiểm tra 100 lượt seed 11 ở checkpoint cuối của lượt 8 màn (100/100, gộp hai seed 197/200) và đường dẫn video quay bằng lệnh chấm. |
