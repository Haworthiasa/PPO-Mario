# Ý tưởng lớn: một mạng mới học nhiều màn với 10 triệu bước

**Trạng thái:** đây là kế hoạch. Chưa có lượt huấn luyện nào của kế hoạch này được chạy. Các con số ở Mục 2 và Mục 3 là số đã đo trên những mạng nhóm đang có.

## 1. Mục tiêu

Huấn luyện **một mạng mới** với ngân sách **10 triệu bước tác tử**, sao cho mạng đó chạm cờ trên 90% ở **càng nhiều màn càng tốt**. Mỗi màn được chấm bằng `mario_ppo/evaluate.py` nguyên bản, lấy mẫu hành động như đề.

Phạm vi:

- Đây là thí nghiệm mở rộng. Thang điểm của đề chỉ tính 1-1, một màn thưởng và một màn ngoài danh sách, nên việc này không cộng điểm.
- Ba tệp trong `submission/` giữ nguyên. Mạng mới không thay tệp nộp nào nếu nhóm không quyết định như vậy.
- Mọi quy tắc ở Mục 3.5 của đề vẫn áp dụng: không sửa `evaluate.py` và phần `eval_mode` của `env.py`; mẹo huấn luyện chỉ nằm trong khối `if not eval_mode:`; đầu vào của mạng chỉ là khung hình đã tiền xử lý.

## 2. Điều đã đo được

### Chi phí cho mỗi màn

| Lượt đã chạy | Bước cho mỗi màn | Kết quả (100 lượt, seed 7) |
|---|---|---|
| Một mạng học 1-1 và 4-1 (`runs/1-1+4-1/from-scratch`) | 1,5 triệu | 99 và 99 |
| Cùng lượt đó ở 2,4 triệu bước | 1,2 triệu | 93 và 96 |
| Cùng lượt đó ở 1,5 triệu bước | 0,75 triệu | 88 và 84 |
| 2-3 học riêng (`runs/2-3/from-scratch-3m`) | 3 triệu | khoảng 83% (sáu checkpoint cuối, hai seed: 998/1200) |

- Ở cùng số bước tính riêng cho từng màn, mạng học chung không nhanh hơn cũng không chậm hơn mạng chuyên (93 và 96 so với 89 và 96). Chi phí vì vậy gần như cộng dồn theo số màn.
- Tốc độ lượt học chung là 132–145 bước mỗi giây. 10 triệu bước mất khoảng 20–21 giờ máy, chưa kể thời gian chấm.
- Suy ra: 10 triệu bước đủ cho khoảng 6 màn dễ cỡ 1-1 và 4-1, hoặc khoảng 3 màn khó cỡ 2-3.

### Kỹ năng không tự mang sang màn khác

| Mạng | Chơi các màn chưa học (20 lượt mỗi màn, seed 2026) | Kết quả ở đâu |
|---|---|---|
| Mạng chung 1-1+4-1, bước 2.904.064 | 0/600 lượt chạm cờ trên 30 màn | `runs/1-1+4-1/from-scratch/eval/all-levels/` |
| Mạng 2-3, bước 2.801.664 | 0/620 lượt chạm cờ trên 31 màn | `runs/2-3/from-scratch-3m/eval/eval-on-<màn>/` |

- Màn gần giống nhất cũng chưa qua: ở 7-3, màn trông gần giống 2-3, mạng 2-3 đi được x trung bình 1178, xa nhất 1867, 0/20. Con số này vẫn cao hơn mọi mạng khác của nhóm ở 7-3 (167 đến 637).
- Ở cảnh lạ, mạng chung gần như chỉ chọn một hành động và đứng kẹt (entropy 0,02–0,03 ở 2-3, 7-3, 6-2). Chi tiết ở `runs/1-1+4-1/from-scratch/eval/all-levels/first-400-steps.txt`.
- Khởi tạo 4-1 từ checkpoint 1-1 chỉ giúp nhanh hơn ở 500 nghìn bước đầu (60 so với 22 trên 100), ngang nhau ở 800–900 nghìn bước (Mục 9 và 11 của `docs/nhat-ky-trong-tu.md`).
- Học lần lượt từng màn làm quên màn cũ trong 100–200 nghìn bước (Mục 10 của nhật ký). Muốn một mạng giữ nhiều màn thì phải trộn các màn trong từng lần cập nhật, như `mario_ppo/train_joint.py` đang làm.

## 3. Hai ý tưởng ban đầu và kết luận

### 3.1 Cho mạng học cách chơi khái quát

Kết luận: chưa tới được với 32 màn và 10 triệu bước.

- Số liệu của nhóm ở Mục 2: hai mạng đều 0 lượt chạm cờ ở mọi màn chưa học.
- Kết quả đã công bố: bộ Procgen dùng 500 màn huấn luyện làm mặc định cho bài toán khái quát hóa, và có trò cần tới 10.000 màn mới hết chênh lệch giữa màn đã học và màn mới. Các lượt PPO chuẩn ở đó dài 200 triệu bước cho mỗi trò.
- Điều làm được trong ngân sách này là một mạng thuộc nhiều màn, không phải một mạng chơi được màn chưa thấy.

### 3.2 Trích một phần màn để luyện theo kỹ năng

Kết luận: làm được về kỹ thuật và có cơ sở, nhưng nó giúp mạng giỏi hơn trong chính màn đó. Nó không tạo ra kỹ năng mang sang màn khác (xem Mục 2).

Đã thử trên 2-3 với checkpoint 2.801.664 (`runs/2-3/from-scratch-3m/eval/failure-analysis-002801664/start-from-snapshot.txt`): lưu trạng thái trình giả lập ở giữa màn, rồi cho mọi lượt sau bắt đầu từ đó.

- Trình giả lập có hàm lưu và nạp trạng thái. Sau khi lưu, mỗi lần `reset()` đưa Mario về đúng điểm đã lưu; vị trí, đồng hồ và phần thưởng đều bình thường.
- **Kết quả phụ thuộc gần như hoàn toàn vào trạng thái được lưu.** Chín trạng thái lưu ở cùng đoạn x ≈ 1500–1600, lấy từ chín lượt chơi khác nhau, cho số lượt chạm cờ trên 40 là: 4, 19, 32, 35, 35, 38, 38, 39, 39. Trong mỗi trạng thái, các lượt hỏng chết ở cùng một chỗ.
- Cách hiểu: trạng thái lưu đóng băng luôn lịch xuất hiện của cá. Mạng thua ở một số tình huống cá cụ thể, và tình huống đó tái tạo lại được để luyện. Dạng mạnh nhất của ý tưởng này là lưu lại chỗ hay chết rồi cho luyện đi luyện lại, cùng họ với cách Salimans và Chen cho tác tử bắt đầu từ các trạng thái giữa chừng ở Montezuma's Revenge.

Ba điều phải xử lý nếu dùng:

1. **Dễ học thuộc.** Mỗi trạng thái lưu chỉ có một lịch quái, nên cần nhiều trạng thái khác nhau. Mỗi trình giả lập chỉ giữ được một trạng thái, nên phải dành riêng một số môi trường cho việc luyện.
2. **Vài bước đầu bị mù chuyển động.** Sau khi nạp, 4 khung hình đầu vào giống hệt nhau. Một trạng thái lưu lúc Mario đang chạy sát mép hố cho 56/60 lượt chết trong 4 bước.
3. **Cần hỏi thầy.** Cách này gọi hàm nội bộ của trình giả lập (`_backup`, `_restore`). Nó chỉ chạy khi huấn luyện, nhưng chưa rõ thầy có coi là mẹo huấn luyện hợp lệ không.

## 4. Kế hoạch

Các bước theo thứ tự. Số bước của từng lượt là đề xuất; nhóm quyết định con số cuối cùng trước khi chạy.

1. **Viết cấu hình thăm dò** `configs/group-multi-pilot.yaml`: giống `configs/group-1-1-and-4-1.yaml`, chỉ khác hai dòng.
   - `levels`: 1-1, 4-1, 2-1, 3-1, 3-2, 5-1, 6-1, 8-1. Tám môi trường, mỗi màn một môi trường (`train_joint.py` yêu cầu số môi trường chia hết cho số màn).
   - `total_steps: 10_000_000`, để tốc độ học và hệ số entropy dùng chung một lịch cho cả kế hoạch.
2. **Chạy lượt thăm dò tới checkpoint 2,0 triệu bước rồi dừng tay** (khoảng 4 giờ, 250 nghìn bước cho mỗi màn). Mã hiện không có khóa "dừng ở bước N"; lượt chạy chỉ tự dừng ở `total_steps`.
   `python -m mario_ppo.train_joint --config configs/group-multi-pilot.yaml run_dir=runs/multi-level/pilot`
3. **Chấm từng màn** ở các checkpoint 1,8 đến 2,0 triệu bước: 100 lượt, seed 7. Xem thêm tỉ lệ chạm cờ khi huấn luyện của từng màn trong `progress.csv`.
4. **Chọn màn giữ lại.** Giữ 4 màn có số lượt chạm cờ cao nhất; nếu bằng nhau thì xét x trung bình. Các màn bị bỏ sẽ bị quên, điều này chấp nhận được.
   - Phương án giữ 6 màn cần 12 môi trường. Phải đo tốc độ với 12 môi trường trước khi chọn phương án này.
5. **Viết cấu hình chính** `configs/group-multi-main.yaml`: giống cấu hình thăm dò, chỉ khác `levels` (các màn được giữ, hai môi trường mỗi màn) và `resume` trỏ tới checkpoint 2,0 triệu bước của lượt thăm dò.
6. **Chạy lượt chính tới 10 triệu bước** (thêm 8 triệu bước, khoảng 16–17 giờ). Bộ đếm bước nối tiếp từ lượt thăm dò nên lịch tốc độ học không bị đặt lại.
   `python -m mario_ppo.train_joint --config configs/group-multi-main.yaml run_dir=runs/multi-level/main`
7. **Chấm và chọn checkpoint.** Chấm mọi checkpoint từ 8 triệu bước trên từng màn (100 lượt seed 7), chọn checkpoint có màn yếu nhất cao nhất. Kiểm lại checkpoint đó bằng 100 lượt seed 11 và bằng đúng lệnh chấm của đề (20 lượt, seed 2026).
8. **Chỉ khi cần và sau khi thầy đồng ý:** nếu một màn kẹt dưới 90% vì vài chỗ chết cố định, viết mẹo luyện từ trạng thái lưu (Mục 3.2), kiểm thử bằng `tests_group/`, rồi mới dùng.
9. **Ghi kết quả** vào `docs/nhat-ky-trong-tu.md` và quyết định có đưa vào báo cáo hay không.

Phân bổ ngân sách theo kế hoạch này:

| Giai đoạn | Số màn | Bước của giai đoạn | Bước cho mỗi màn được giữ (cộng dồn) |
|---|---|---|---|
| Thăm dò | 8 | 2 triệu | 0,25 triệu |
| Chính | 4 | 8 triệu | 2,25 triệu |

Lý do chọn màn cho lượt thăm dò:

- 1-1 và 4-1 đã biết là học được với 1,5 triệu bước mỗi màn.
- Sáu màn mới (2-1, 3-1, 3-2, 5-1, 6-1, 8-1) là các màn mặt đất, ngày hoặc đêm. Đây là phỏng đoán theo hình dạng màn; chưa có số liệu về độ khó của chúng. Lượt thăm dò là thứ quyết định.
- Tránh tám màn lâu đài (ba màn trong đó là mê cung), hai màn nước và các màn x-3. Màn 2-3 đã cho thấy màn x-3 tốn gấp đôi; tác giả kho mã PPO Mario công khai của uvipen cho biết phải thử khoảng 70 lần mới qua được 1-3.

## 5. Tiêu chí thành công

- Một màn được tính là đạt khi cùng một checkpoint chạm cờ ít nhất 90/100 ở seed 7 và ít nhất 90/100 ở seed 11.
- Kỳ vọng: 4 màn đạt là khả năng cao, vì hai màn đã làm được ở lượt 3 triệu bước; 6 màn là có thể; 8 màn thì khó. Đây là ước lượng, chưa có số liệu cho trường hợp nhiều hơn hai màn.

## 6. Rủi ro và điều chưa biết

- **Mỗi màn chỉ có một môi trường ở lượt thăm dò.** Mỗi lần cập nhật chỉ có khoảng 2 lượt chơi cho mỗi màn. Lượt hai màn đã chạy dùng 4 môi trường mỗi màn, nên chưa biết tín hiệu có đủ rõ không.
- **Sức chứa của mạng.** Mạng hiện tại chứa được hai màn ở 99%. Chưa biết giới hạn khi số màn tăng.
- **Một phần ngân sách bị bỏ.** Bốn màn bị loại sau lượt thăm dò dùng mất khoảng 1 triệu bước.
- **Chỉ có một seed.** Mọi kết luận sẽ dựa trên một lượt chạy; chênh lệch vài điểm phần trăm không đủ để kết luận.
- **Thời gian.** Khoảng 20–21 giờ máy cho huấn luyện, cộng thời gian chấm. Báo cáo và phần trình bày (30 điểm) chưa viết và không được chậm vì việc này.

## 7. Việc cần hỏi thầy

- Mẹo huấn luyện bắt đầu lượt từ trạng thái trình giả lập đã lưu có hợp lệ theo Mục 3.5 của đề không.

## 8. Nguồn

- Cobbe và cộng sự, "Leveraging Procedural Generation to Benchmark Reinforcement Learning" (Procgen): https://arxiv.org/abs/1912.01588
- OpenAI, "Procgen Benchmark": https://openai.com/index/procgen-benchmark/
- Salimans và Chen, "Learning Montezuma's Revenge from a Single Demonstration": https://arxiv.org/abs/1812.03381
- uvipen, Super-mario-bros-PPO-pytorch: https://github.com/uvipen/Super-mario-bros-PPO-pytorch

---

# v2: mở rộng mạng 1-1+4-1, thêm ba màn mới cùng lúc

**Trạng thái (21/11/2026):** kế hoạch. Bản này thay Mục 4, 5 và 6 ở trên; Mục 1, 2, 3 và 7 vẫn dùng. Một lượt thử 372.736 bước đã chạy (Mục v2.3); checkpoint của nó đã xóa, chỉ giữ số liệu. Ngoài lượt đó chưa có lượt huấn luyện nào của kế hoạch này.

## v2.1 Hướng được chọn

Nối tiếp từ checkpoint mạng chung 1-1+4-1 (bước 2.904.064), cho sáu màn mới vào **cùng lúc** để thăm dò, giữ ba màn học nhanh nhất, rồi chạy tới bước tổng 9.998.336.

- Mục tiêu: 5 màn trên 90% ở cùng một checkpoint. Mức sàn: 4 màn.
- Ngân sách: 10 triệu bước cộng dồn, tính cả 2.904.064 bước đã tạo ra checkpoint xuất phát. Còn 7.094.272 bước.
- Khác bản v1 ở trên: không học lại từ đầu.
- Khác bản V1 trong `codexbigidea.md`: các màn mới vào cùng lúc thay vì lần lượt, và màn được chọn bằng số đo của một giai đoạn thăm dò. Bản V2 của tệp đó được so ở Mục v2.8.

Ba lý do, số liệu ở hai mục sau:

1. Hai màn đã có sẵn ở 99/100, nên không phải trả lại 2,9 triệu bước cho chúng.
2. Rủi ro lớn nhất của việc nối tiếp đã được thử: 1-1 và 4-1 tụt nhưng không sập.
3. Mỗi màn mới được giữ nhận 1,67 triệu bước của riêng nó, trong khi một màn cỡ 4-1 đã đo là cần 1,0–1,3 triệu.

## v2.2 Số liệu từ các lượt đã chạy

Đọc từ `progress.csv` của từng lượt. "Bước của riêng màn" là số bước mà màn đó nhận được. Tỉ lệ chạm cờ ở đây là tỉ lệ khi huấn luyện (100 lượt gần nhất).

| Lượt | Tỉ lệ chạm cờ vượt 0,9 khi nào | Điều kiện |
|---|---|---|
| 4-1 học từ đầu | 0,9–1,1 triệu bước của riêng màn (0,92) | lr giảm về 0 trong 1,2 triệu bước |
| 1-1 trong lượt học chung | 0,9–1,1 triệu (0,91) | lr giảm về 0 trong 3 triệu bước |
| 4-1 trong lượt học chung | 1,1–1,3 triệu (0,94) | như trên |
| 4-1 khởi tạo từ mạng 1-1 | không vượt, đứng ở 0,82–0,85 | lr giữ ở 0,9–1,2e-4, hệ số entropy cố định 0,01 |
| 2-3 học từ đầu | không vượt trong 3 triệu bước (0,85) | lr giảm về 0 trong 3 triệu bước |

- **Khởi tạo từ mạng đã học giúp nhanh ở đầu.** Sau 75–100 nghìn bước ở 4-1, x trung bình là 2065 khi khởi tạo từ mạng 1-1 và 876 khi học từ đầu.
- **Các lượt vượt 0,9 đều vượt khi lr đã xuống dưới 7e-5.** Lịch của v2 xuống dưới 7e-5 từ khoảng bước 6,5 triệu, nên còn 3,5 triệu bước cuối cho việc ổn định.
- **Giai đoạn đầu không báo trước được màn khó.** Ở 400–500 nghìn bước, tỉ lệ chạm cờ của 2-3 là 0,20, ngang 4-1 (0,19) và 1-1 (0,24). Chỉ sau 500 nghìn bước 2-3 mới tụt lại. Vì vậy kế hoạch cần thêm một mốc quyết định muộn (Bước 5).

## v2.3 Lượt thử nối tiếp (đã chạy)

Cấu hình: `configs/group-1-1-and-4-1.yaml`, nối tiếp checkpoint 2.904.064, chỉ đổi `levels` thành 8 màn (1-1, 4-1, 2-1, 3-1, 3-2, 5-1, 6-1, 8-1, mỗi màn một môi trường) và `total_steps` thành 9.998.336. Tại lúc nối tiếp, lr là 1,42e-4 và hệ số entropy là 0,0077 (ở cuối lượt học chung là 6,4e-6 và 0,0023).

Lượt này dự định dừng sau khoảng 66 nghìn bước. Lệnh dừng không có tác dụng, nên nó chạy 372.736 bước (tới bước tổng 3.276.800) rồi mới được dừng tay. Số bước này chưa được nhóm duyệt trước.

- Số liệu được giữ: nhật ký huấn luyện và bảng tóm tắt ở `runs/multi-level/restart-test/`.
- Checkpoint, video và tệp kết quả chấm của lượt thử đã xóa theo quyết định của nhóm, nên lượt thử không nối tiếp được. Giai đoạn thăm dò ở Mục v2.4 chạy lại từ checkpoint 2.904.064.
- 372.736 bước này không nằm trên đường huấn luyện tạo ra checkpoint cuối. Kế hoạch dưới đây không trừ chúng vào ngân sách 10 triệu; khi báo cáo cần ghi riêng.

**Hai màn cũ** (`evaluate.py` nguyên bản, 100 lượt, seed 7):

| Bước sau khi nối tiếp | 1-1 | 4-1 |
|---|---|---|
| 0 | 99 | 99 |
| 24.576 | 98 | 99 |
| 106.496 | 88 | 98 |
| 208.896 | 75 | 83 |
| 372.736 | 80 | 89 |

- Hai màn cũ mất 10–19 điểm so với lúc xuất phát nhưng không sập. Hai lần chấm cuối là 75 và 83, rồi 80 và 89; chênh lệch 5–6 điểm này nằm trong sai số của phép chấm 100 lượt (khoảng ±5 điểm), nên chưa thể nói hai màn đang hồi phục.
- Cả 91 lần cập nhật đều bị `target_kl` dừng sớm, sau 32–43 trên 160 bước gradient (ở lượt học chung từ đầu, mức thấp nhất là 117). approxKL quanh 0,02–0,035, cao nhất 0,089. Mức thay đổi của chính sách đang do `target_kl` giới hạn.
- Vì chạy ít bước gradient hơn, tốc độ lên 206 bước mỗi giây (lượt học chung từ đầu: 132–145).

**Sáu màn mới** (x trung bình khi huấn luyện; mỗi màn nhận 46.592 bước):

| Màn | Sau 28.672 bước | Sau 372.736 bước | Tỉ lệ chạm cờ |
|---|---|---|---|
| 3-2 | 752 | 2214 | 0,01 |
| 6-1 | 443 | 1640 | 0 |
| 8-1 | 785 | 1441 | 0 |
| 2-1 | 566 | 1133 | 0 |
| 5-1 | 566 | 919 | 0 |
| 3-1 | 504 | 681 | 0 |

- 3-2 tiến nhanh nhất; 3-1 và 5-1 chậm nhất; 2-1 gần như đứng yên quanh 1030–1130 trong 200 nghìn bước cuối.
- Chưa đo chiều dài từng màn, nên x chưa so được chính xác giữa các màn.

Giới hạn: một lượt, một seed. Lượt thử chưa cho biết 1-1 và 4-1 có trở lại trên 90% khi lr giảm hay không.

## v2.4 Kế hoạch

Các bước theo thứ tự. Số bước là đề xuất; nhóm quyết định con số cuối cùng trước khi chạy.

1. **Viết cấu hình thăm dò** `configs/group-multi-probe.yaml`: giống `configs/group-1-1-and-4-1.yaml`, chỉ khác ba khóa.
   - `levels`: 1-1, 4-1, 2-1, 3-1, 3-2, 5-1, 6-1, 8-1.
   - `algo.total_steps: 9_998_336`. Lượt chạy chỉ tự dừng ở cuối rollout đầu tiên đạt `total_steps`; 9.998.336 là mốc cuối rollout lớn nhất không vượt 10 triệu.
   - `resume: runs/1-1+4-1/from-scratch/checkpoints/step_002904064.pt`.
   - `lr` và lịch entropy giữ nguyên. Bước này không cần sửa mã.
2. **Chạy thăm dò tới checkpoint 3.702.784 rồi dừng tay.** Khi đó mỗi màn mới có 99.840 bước. Cấu hình và seed trùng với lượt thử, nên 372.736 bước đầu dự kiến cho kết quả gần với Mục v2.3.
   `python -m mario_ppo.train_joint --config configs/group-multi-probe.yaml run_dir=runs/multi-level/probe`
3. **Chấm và chọn màn** ở checkpoint 3.702.784: 100 lượt, seed 7, cho cả tám màn.
   - Điều kiện đi tiếp: 1-1 và 4-1 đều từ 70/100 trở lên. Nếu không, chuyển sang phương án lùi (Mục v2.5).
   - Giữ ba màn mới, xếp theo số lượt chạm cờ, bằng nhau thì theo x trung bình. Theo số liệu hiện có, 3-2 gần như chắc được giữ; hai suất còn lại nằm giữa 6-1, 8-1 và 2-1.
   - Nếu màn xếp thứ ba có x không tăng trong 200 nghìn bước cuối, chỉ giữ hai màn mới với 3 môi trường mỗi màn.
4. **Sửa `mario_ppo/train_joint.py`** để khai được số môi trường của từng màn. Hàm `assign_levels` hiện đòi chia đều và không cho lặp màn. Sửa bài kiểm tra trong `tests_group/test_train_joint.py` cho khớp. Viết `configs/group-multi-main.yaml`: 1-1 và 4-1 mỗi màn 1 môi trường, ba màn mới mỗi màn 2 môi trường, `resume` trỏ tới checkpoint 3.702.784.
5. **Chạy lượt chính** (`run_dir=runs/multi-level/main`) và chấm ở hai loại mốc.
   - Trong lúc chạy: chấm 1-1 và 4-1 (100 lượt, seed 7) ở mỗi triệu bước. Nếu một màn dưới 70/100 ở hai lần chấm liên tiếp, dừng và chuyển sang phương án lùi.
   - Ở checkpoint 7.503.872 (lr còn 5,0e-5, mỗi màn mới đã có 1,05 triệu bước): chấm cả năm màn. Một màn mới còn dưới 70/100 thì khó vượt 90 trong 0,62 triệu bước còn lại của nó; ở 2-3, tỉ lệ chỉ tăng từ 0,77 lên 0,85 trong 0,6 triệu bước cuối.
   - Nếu có màn dưới 70 **và** có màn khác đang ở 80–92: dừng, bỏ màn dưới 70, chuyển 2 môi trường của nó cho hai màn thấp nhất còn lại, rồi nối tiếp vào `runs/multi-level/final`.
   - Các trường hợp khác: để lượt chạy đi tiếp, không đổi gì.
6. **Chạy tới bước 9.998.336.**
7. **Chọn checkpoint và xác nhận.** Chấm mọi checkpoint từ 9 triệu bước trên từng màn (100 lượt, seed 7). Chọn checkpoint có nhiều màn đạt 90/100 nhất; bằng nhau thì xét màn yếu nhất. Kiểm lại bằng 100 lượt seed 11 và bằng lệnh chấm của đề (20 lượt, seed 2026).
8. **Khảo sát lại 32 màn** bằng checkpoint được chọn (20 lượt, seed 2026) và so với `runs/1-1+4-1/from-scratch/eval/all-levels/summary.csv`. Phép so này cho biết quãng đường trên các màn chưa học thay đổi thế nào. Ba màn bị loại sau thăm dò phải ghi rõ là đã được học khoảng 100 nghìn bước.
9. **Ghi kết quả** vào `docs/nhat-ky-trong-tu.md`.

Phân bổ ngân sách:

| Giai đoạn | Bước tổng | 8 môi trường | lr |
|---|---|---|---|
| Thăm dò | 2.904.064 → 3.702.784 | 8 màn, mỗi màn 1 | 1,42e-4 → 1,26e-4 |
| Chính | 3.702.784 → 7.503.872 | 1 + 1 + 2 + 2 + 2 | 1,26e-4 → 5,0e-5 |
| Cuối | 7.503.872 → 9.998.336 | như trên, hoặc đổi theo Bước 5 | 5,0e-5 → 0 |

| Màn | Bước của riêng màn sau checkpoint 2.904.064 |
|---|---|
| 1-1 và 4-1 | 886.784 mỗi màn (cộng 1,45 triệu đã có) |
| Mỗi màn mới được giữ | 1.673.728 |
| Mỗi màn bị loại sau thăm dò | 99.840 |

Thời gian máy: khoảng 9,5 giờ nếu tốc độ giữ ở 206 bước mỗi giây, khoảng 15 giờ nếu về 132. Chưa kể thời gian chấm.

## v2.5 Phương án lùi

Dùng khi 1-1 hoặc 4-1 không giữ được theo điều kiện ở Bước 3 và Bước 5.

- Nối tiếp từ checkpoint gần nhất mà cả hai màn còn từ 70/100 trở lên.
- Phân bổ 2 + 2 + 2 + 2: 1-1, 4-1 và hai màn mới tốt nhất. Mã hiện tại chạy được phân bổ này.
- `total_steps` giữ 9.998.336. Mục tiêu khi đó là 4 màn.

## v2.6 Tiêu chí thành công

- Giữ tiêu chí của Mục 5: một màn đạt khi cùng một checkpoint chạm cờ ít nhất 90/100 ở seed 7 và ít nhất 90/100 ở seed 11.
- Ước lượng: 4 đến 5 màn. Đây là ước lượng từ chi phí mỗi màn ở Mục v2.2, chưa có lượt nào của nhóm kiểm chứng.

## v2.7 Rủi ro và điều chưa biết

- **1-1 và 4-1 có thể không trở lại trên 90%.** Mỗi màn chỉ còn 1/8 dữ liệu, và lượt thử mới đo tới 80 và 89. Bước 5 theo dõi việc này; phương án lùi tăng phần dữ liệu của hai màn lên 1/4.
- **Màn khó kiểu 2-3 không lộ ra sớm.** Thăm dò chỉ loại được màn tiến chậm ngay từ đầu. Mốc 7.503.872 là chỗ bắt loại còn lại.
- **Xếp hạng thăm dò dựa trên x.** Các màn dài ngắn khác nhau và chưa đo chiều dài.
- **`target_kl` dừng sớm mọi lần cập nhật.** Mỗi lần chỉ chạy khoảng một phần tư số bước gradient của cấu hình. Chưa biết việc này làm màn mới học chậm đi bao nhiêu tính theo bước môi trường.
- **Một seed.** Chênh lệch vài điểm phần trăm chưa đủ để kết luận.
- **Thời gian.** Báo cáo và phần trình bày (30 điểm) chưa viết và không được chậm vì việc này.

## v2.8 Các hướng không chọn

| Hướng | Lý do không chọn |
|---|---|
| Học lại từ đầu (v1) | 21 giờ máy; mốc quyết định đầu tiên sau 4 giờ; trả lại 2,9 triệu bước cho hai màn đã có. |
| Bản V2 trong `codexbigidea.md`: 1-1, 4-1, 3-2, 5-1, mỗi màn 2 môi trường; lr khoảng 5,0e-5 lúc nối tiếp | Tối đa 4 màn nếu không sửa mã. 5-1 xếp thứ năm trong sáu màn ở lượt thử. Lượt thử cho thấy hai màn cũ chịu được lr 1,42e-4 với 1/8 dữ liệu, còn tốc độ học một màn mới ở lr dưới 5e-5 thì chưa có số đo. Phân bổ 2 + 2 + 2 + 2 của bản này trùng với phương án lùi ở Mục v2.5. |
| Thêm màn lần lượt (bản V1 trong `codexbigidea.md`) | Với lịch lr giảm tuyến tính tới 10 triệu bước, màn thứ tư bắt đầu khi lr còn 1,0e-4, màn thứ năm khi còn 5e-5; mỗi màn vào sau có ít bước ở lr cao hơn màn vào trước. |
| 16 môi trường để giữ 6 màn | Mỗi màn mới chỉ còn 1,33 triệu bước, sát mức 1,0–1,3 triệu đã đo; 16 môi trường là cấu hình nhóm chưa từng chạy. |
| Luyện từ trạng thái lưu giữa màn | Chỉ xét khi một màn kẹt ở 80–89% vì một chỗ chết cố định, và sau khi thầy đồng ý (Mục 3.2 và Mục 7). |
