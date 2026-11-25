# Báo cáo màn 2-3

Cập nhật lần cuối: 24/11/2026, 07:45. Màn 2-3 chỉ có hai lượt một màn; lượt 8 màn không huấn luyện màn này.

Quy ước: **bước** là bước tác tử. **Số chấm** là kết quả của `evaluate.py` (100 lượt seed 7 để ra quyết định, 20 lượt seed 2026 là đúng lệnh chấm của đề). **Số lúc huấn luyện** là tỉ lệ chạm cờ của 100 lượt huấn luyện gần nhất trong `progress.csv`; hai loại số có thể lệch nhau hơn 10 điểm. Cờ của 2-3 ở x = 3593; ngưỡng vượt màn là 0,8.

## 1. Kết quả tốt nhất hiện có

| Mạng | Checkpoint | 100 lượt, seed 7 | 100 lượt, seed 11 | 20 lượt, seed 2026 |
|---|---|---|---|---|
| Chỉ học 2-3, 3 triệu bước (**bản nộp**, `submission/2-3.pt`) | `runs/2-3/from-scratch-3m`, bước 2.801.664 | 86/100 | 87/100 | 18/20 |
| Chỉ học 2-3, 1,2 triệu bước | `runs/2-3/from-scratch`, bước 1.200.128 | 55/100 | chưa chấm | 9/20 |

Màn 2-3 vượt ngưỡng ở bản nộp, nhưng sát ngưỡng hơn 1-1 và 4-1: gộp hai seed là 173/200. Chưa có mạng nhiều màn nào học 2-3.

## 2. Các lượt chạy có huấn luyện 2-3

Mọi lượt đều dùng seed 1, 8 môi trường × 512 bước, γ = 0,9, bộ hành động `simple`.

| Lượt chạy | Số màn | Tổng bước | Bước riêng trên 2-3 | Tốc độ học | Chạm cờ lúc huấn luyện, cuối lượt |
|---|---|---|---|---|---|
| `runs/2-3/from-scratch` | 1 | 1.200.128 | 1.200.128 | 2e-4 giảm về 0 | 52% |
| `runs/2-3/from-scratch-3m` | 1 | 3.002.368 | 3.002.368 | 2e-4 giảm về 0 | 88% |

- Hai tệp `config.yaml` chỉ khác `total_steps` (1.200.000 và 3.000.000) và `run_dir`. Lịch tốc độ học và lịch hệ số entropy (0,01 giảm về 0,002) đều tính theo `total_steps`, nên ở lượt 3 triệu bước cả hai lịch giảm chậm hơn.
- Cấu hình giống lượt 4-1 học từ đầu (`configs/group-4-1.yaml`), chỉ khác màn.

## 3. Đường cong học

![Tỉ lệ chạm cờ lúc huấn luyện của màn 2-3 qua các lượt chạy](figures/2-3-training.png)

Hình vẽ bằng `python experiments/level_curves.py 2-3`. Cả hai lượt đều là lượt một màn nên hai khung của hình giống nhau.

- **Lượt 3 triệu bước chạm 80% lần đầu ở 1,94 triệu bước.** Nó chạm 50% ở 0,80 triệu bước, cao nhất 91% ở 2,99 triệu bước và kết thúc ở 88%. Trong 50 dòng cuối của `progress.csv` (khoảng 200 nghìn bước) số này nằm trong khoảng 78–91%.
- **Lượt 1,2 triệu bước không tới ngưỡng.** Nó kết thúc ở 52%, cũng là mức cao nhất của lượt, và vẫn đang tăng khi tốc độ học về 0.
- **Ở cùng số bước, lượt 3 triệu bước cao hơn.** Tại 0,5 triệu bước: 30% so với 13%. Tại 1,0 triệu bước: 66% so với 26%.

Giới hạn: mỗi lượt chỉ một seed. Hai lượt chỉ khác một dòng cấu hình, nhưng dòng đó kéo dài cả lịch tốc độ học lẫn lịch entropy, nên chưa tách được phần của số bước và phần của hai lịch. Các nhận xét trên là quan sát, chưa phải kết luận về nguyên nhân.

## 4. Số chấm theo checkpoint

### Lượt 1,2 triệu bước (`runs/2-3/from-scratch`)

| Bước | 100k | 200k | 300k | 400k | 500k | 600k | 700k | 800k | 900k | 1,0M | 1,1M | 1,2M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chạm cờ /100 (seed 7) | 0 | 0 | 0 | 1 | 16 | 20 | 35 | 40 | 41 | 46 | 51 | **55** |
| Chạm cờ /20 (seed 2026) | 0 | 0 | 0 | 1 | 4 | 3 | 8 | 10 | 10 | 6 | 8 | **9** |
| x trung bình (seed 7) | 1192 | 1473 | 1774 | 2132 | 2509 | 2498 | 2772 | 2825 | 2799 | 2961 | 2947 | 3062 |

### Lượt 3 triệu bước (`runs/2-3/from-scratch-3m`)

| Bước | 0,5M | 1,0M | 1,5M | 2,0M | 2,3M | 2,4M | 2,5M | 2,6M | 2,7M | 2,8M | 2,9M | 3,0M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chạm cờ /100 (seed 7) | 30 | 58 | 72 | 73 | 80 | 79 | 82 | 81 | 79 | **86** | 80 | 80 |
| Chạm cờ /100 (seed 11) | – | – | – | – | 80 | 82 | 86 | 81 | 84 | **87** | 84 | 88 |
| Chạm cờ /20 (seed 2026) | 2 | 13 | 14 | 14 | 16 | 17 | 18 | 16 | 14 | **18** | 17 | 18 |
| x trung bình (seed 7) | 2573 | 2897 | 3199 | 3228 | 3325 | 3270 | 3326 | 3296 | 3237 | 3358 | 3269 | 3331 |

Bảng đủ 30 checkpoint ở Mục 13 của `docs/nhat-ky-trong-tu.md`.

- Checkpoint 2.801.664 được chọn nộp vì cao nhất khi gộp hai seed (173/200) và đạt 18/20 theo lệnh chấm. Tệp `submission/ket-qua-2-3.json` ghi 18/20; hai lượt hỏng dừng ở x = 1989 và x = 3179.
- Hai checkpoint cạnh nhau có thể chênh nhiều: 53 rồi 36 ở 0,8–0,9 triệu bước, 72 rồi 57 ở 1,5–1,7 triệu bước. Với 20 lượt, checkpoint 2.703.360 được 14/20 trong khi hai checkpoint cạnh nó được 16 và 18.
- Ở cùng bước, lượt 3 triệu bước cao hơn lượt 1,2 triệu bước: 58 so với 46 ở bước 1.003.520, 61 so với 55 ở bước 1.200.128.

## 5. Mario hỏng ở đâu

### Bản nộp (bước 2.801.664, 100 lượt seed 7, 14 lượt hỏng: 13 rơi hố, 1 chạm địch)

| Vị trí x | Số lượt | Chuyện xảy ra |
|---|---|---|
| 1331–1333 | 4 | rơi hố; ở seed 11 không có lượt nào chết ở đây và khi chọn hành động xác suất cao nhất cũng không, nên là do lấy mẫu |
| 1577 | 1 | rơi hố |
| 1797 | 1 | rơi hố |
| 1956–2027 | 6 | chỗ nhảy từ bệ xanh thấp lên cầu; 5 lượt rơi hố, 1 lượt chạm địch (x = 1979) |
| 2873 | 1 | rơi hố |
| 3315 | 1 | rơi hố |

Ở seed 11 có 13 lượt hỏng (10 rơi hố, 3 chạm địch), trong đó 7 lượt dừng ở x 1859–1989. Gộp hai seed: 27 lượt hỏng trên 200, 13 lượt dừng ở x 1800–2060.

| Cách chọn hành động (100 lượt, seed 7) | Chạm cờ |
|---|---|
| Lấy mẫu, như đề chấm (T = 1) | 86 |
| T = 0,75 | 86 |
| T = 0,5 | 88 |
| T = 0,25 | 94 |
| Luôn chọn xác suất cao nhất | 96 |

- **Lấy mẫu làm mất khoảng 10 điểm.** 14,2% số hành động lấy mẫu khác hành động xác suất cao nhất (4005 trên 28.255 bước).
- **Một điểm chết cố định ở x = 1989.** Khi luôn chọn xác suất cao nhất, cả 4 lượt hỏng đều rơi hố ở x = 1989 đúng bước thứ 166.
- **Điểm chết đó đi cùng việc đạp trúng cá.** Trong 200 lượt lấy mẫu: có đạp trúng cá ở x 1800–2000 thì 11 trên 25 lượt chết ở x 1800–2060; không thì 2 trên 167. Đây là tương quan; quan hệ nhân quả chưa được kiểm chứng riêng.
- **Kết quả phụ thuộc thời điểm tới đoạn đó.** Cho Mario bắt đầu từ chín ảnh chụp trạng thái trò chơi ở x≈1500–1600 (40 lượt mỗi ảnh, lấy mẫu): số lượt chạm cờ đi từ 4/40 tới 39/40 tùy ảnh chụp (`start-from-snapshot.txt`).
- **Mạng dùng nút "trái" ở vài chỗ cố định.** 220 lần bấm trong 89 trên 100 lượt, dồn ở x 3300–3399, 2400–2499 và 3100–3199. Cấm nút này thì còn 60/100 (seed 7) và 68/100 (seed 11), phần mất chủ yếu là chạm địch ở x 2400–2499 và 3200–3299 (`left-action.txt`).
- Mọi tệp và ảnh ở `runs/2-3/from-scratch-3m/eval/failure-analysis-002801664/`.

### Checkpoint 1,2 triệu bước (`runs/2-3/from-scratch`, 100 lượt seed 7, 45 lượt hỏng)

| Đoạn x | Chết / số lượt tới được, bước 1.200.128 | Chết / số lượt tới được, bản nộp |
|---|---|---|
| 1950–2030 | 13/94 | 6/94 |
| 2442–2562 | 9/81 | 0/88 |
| 2797–2918 | 7/72 | 1/88 |
| 3193–3320 | 9/64 | 1/87 |

- Chọn hành động xác suất cao nhất ở checkpoint này đạt 76/100, so với 55/100 khi lấy mẫu. Khi đó các lượt hỏng dồn ở x = 1081, 1483, 1979–1985, 2536–2557, 3304–3305.
- Trong 20 lượt seed 2026: 10 lượt rơi hố, 1 lượt chạm địch, 9 lượt chạm cờ (`eval/death-causes-001200128/look.txt`).
- Từ 1,2 lên 2,8 triệu bước, ba đoạn sau gần hết chết; đoạn x 1950–2030 giảm ít nhất.

## 6. Mạng không học 2-3 thì chơi 2-3 thế nào

| Mạng | Chạm cờ ở 2-3 (20 lượt, seed 2026) | x trung bình | x xa nhất |
|---|---|---|---|
| Chỉ học 1-1 (`runs/1-1/run-1`, bước 1.200.128) | 0/20 | 843 | 1762 |
| Chỉ học 4-1 (`runs/4-1/from-scratch`, bước 1.200.128) | 0/20 | 1024 | 1792 |
| Một mạng 2 màn 1-1 và 4-1, bước 1.503.232 | 0/20 | 172 | 608 |
| Một mạng 2 màn 1-1 và 4-1, bước 1.900.544 | 0/20 | 244 | 873 |
| Một mạng 2 màn 1-1 và 4-1, bước 2.904.064 | 0/20 | 192 | 650 |
| Một mạng 8 màn (`runs/8-levels/from-scratch-10m`, bước 9.998.336) | 0/20 | 259 | 1081 |

- Mạng chỉ học 1-1: 10 lượt chạm địch trước x = 832, 10 lượt rơi hố ở x 1071–1762. Mạng chỉ học 4-1: 15 lượt rơi hố, 5 lượt chạm địch.
- Không mạng nào chạm cờ. Hai mạng nhiều màn đi được ngắn hơn hẳn hai mạng một màn (x trung bình 172–259 so với 843–1024). Ở mạng 2 màn, x trung vị của cả ba checkpoint là 146, tức hơn nửa số lượt dừng ở x = 146 (`runs/1-1+4-1/from-scratch/eval/all-levels/unseen-by-checkpoint.txt`). Ở mạng 8 màn, 16 trên 20 lượt dừng ở x = 146, mỗi lượt khoảng 1500 bước (`runs/8-levels/from-scratch-10m/eval/inside-network-009998336/unseen-levels/2-3.json`). Chưa xem khung hình để biết vì sao.

## 7. Điều chưa biết

- Đạp trúng cá có phải là nguyên nhân làm rơi hố ở x = 1989 hay chỉ đi cùng nhau.
- Hai cách chưa thử: cho một phần môi trường huấn luyện bắt đầu ngay trước x≈1800 để gặp đoạn đó nhiều hơn; hạ `ent_coef_final` xuống dưới 0,002.
- Học 2-3 cùng các màn khác trong một mạng thì nhanh hơn hay chậm hơn tính theo bước riêng (ở 1-1 và 4-1 đã có số; ở 2-3 chưa có lượt nào).
- Vì sao hai mạng nhiều màn dừng ở x = 146 trên 2-3 trong khi hai mạng một màn đi qua được chỗ đó.
- Mọi kết quả mới có một seed huấn luyện.

Các hướng mở rộng chung cho mọi màn ở `docs/extension-directions.md`.

## 8. Nhật ký cập nhật

| Ngày | Thay đổi |
|---|---|
| 24/11/2026 | Tạo báo cáo từ hai lượt một màn đã có. |
| 24/11/2026 | Thêm số của mạng 8 màn (checkpoint cuối, chưa học 2-3) vào Mục 6, lấy từ `runs/8-levels/from-scratch-10m/eval/inside-network-009998336/unseen-levels/2-3.json`. |
