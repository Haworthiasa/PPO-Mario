# Rà soát cấu hình huấn luyện: đã tối ưu chưa, nên sửa gì

Ghi ngày 26/11/2026. Bản này không huấn luyện thêm bước nào. Số liệu đọc lại từ `progress.csv` và `allocation.csv` của bảy lượt chạy bằng `experiments/config_review.py`. Các con số trong tài liệu tham khảo được đọc trực tiếp từ những trang liệt kê ở Mục 9. Chỗ nào là suy luận hoặc ước tính đều ghi rõ.

Câu hỏi: các giá trị trong `configs/group-*.yaml` đã tối ưu chưa, nên sửa chỗ nào, và nên làm thêm gì.

## Tóm tắt

1. **Chưa có cách nào chứng minh cấu hình là "tối ưu". Tuy vậy, cấu hình hiện tại hợp lý và không có lỗi.**
   - Nó gần như trùng cấu hình PPO cho Mario của uvipen: 8 môi trường × 512 bước, 16 mini-batch, 10 epoch, γ = 0,9, λ = 1,0, entropy 0,01, cắt gradient 0,5.
   - Nhóm thêm ba thứ: tốc độ học giảm về 0, hệ số entropy giảm dần, và `target_kl`.
   - Ở cả bảy lượt, approxKL, tỉ lệ cắt và explained variance đều nằm trong vùng bình thường.
2. **Cấu hình không còn là chỗ quyết định điểm.**
   - Ba bản nộp đã qua ngưỡng.
   - Nếu chấm lại trên một máy khác, xác suất rớt dưới 16/20 là khoảng 0,1% với 4-1, 6% với 1-1 và 12% với 2-3. Tính cả ba, kỳ vọng chỉ mất khoảng 0,2 điểm.
   - 30 điểm còn để ngỏ nằm ở báo cáo (25) và trình bày (5).
3. **Có ba chỗ mà số đo cho thấy cấu hình chưa khớp với cách nó được dùng:**
   - **"10 epoch" chỉ đúng khi học một màn.** Ở lượt 8 màn, `target_kl` 0,03 dừng sớm 72% số lần cập nhật. Trung bình mỗi lần chỉ chạy 6,3 epoch, và ở đoạn 2–4 triệu bước chỉ còn 3,8 epoch. CleanRL, SB3 Zoo và Procgen dùng 3–4 epoch; chỉ uvipen dùng 10.
   - **γ = 0,9 chỉ nhìn xa khoảng 10 bước (0,67 giây).** Critic đo được chỉ thấy trước 7–15 bước. Ngay cả một critic hoàn hảo với γ = 0,9 cũng chỉ thấy tới khoảng 22 bước. Atari dùng γ = 0,99 với cùng frame skip. Đây là ứng viên tốt nhất cho thí nghiệm ở Mục 4 của báo cáo.
   - **Cách chia bước theo tỉ lệ hỏng chia sai chỗ.** Nó dồn 2,04 triệu bước (20% tổng số) vào màn 2-1, và màn đó chỉ đạt 3,5%.
4. **Phần lớn thời gian chạy nằm ở bước cập nhật mạng.**
   - Bước cập nhật chiếm 41–58% thời gian mỗi vòng thu và cập nhật.
   - WSL nhìn thấy card GTX 1650 Ti, nhưng torch đang là bản CPU.
   - Nếu dùng GPU, mỗi vòng có thể nhanh khoảng 1,6–2 lần (ước tính, chưa đo).
5. **Hai việc không nên làm:**
   - Không đổi sang IMPALA-CNN khi còn chạy trên CPU: mỗi bước gradient chậm hơn khoảng 10–12 lần (đo được).
   - Không hạ "nhiệt độ" của chính sách sau khi huấn luyện xong: đề yêu cầu chấm bằng cách lấy mẫu "như khi huấn luyện".

## 1. So với các cấu hình đã công bố

| Tham số | Nhóm (2-3, 1-1+4-1, 8 màn) | uvipen, PPO cho Mario | CleanRL và SB3 Zoo, Atari | Procgen (Cobbe 2020) |
|---|---|---|---|---|
| Môi trường × bước mỗi đoạn thu | 8 × 512 = 4.096 | 8 × 512 = 4.096 | 8 × 128 = 1.024 | 64 × 256 mỗi worker |
| Mini-batch | 16 mini-batch × 256 mẫu | 16 mini-batch × 256 mẫu | 4 × 256 | 8 |
| Epoch | 10, dừng sớm khi KL > 0,03 | 10 | 4 | 3 |
| Tốc độ học | 2e-4, giảm tuyến tính về 0 | 1e-4, cố định (chỉnh riêng từng màn) | 2,5e-4, giảm tuyến tính về 0 | 5e-4 |
| γ | **0,9** | 0,9 | 0,99 | 0,999 |
| λ (GAE) | **1,0** | 1,0 | 0,95 | 0,95 |
| Cắt ε | 0,15 | 0,2 | 0,1 | 0,2 |
| Hệ số entropy | 0,01 → 0,002 | 0,01 | 0,01 | 0,01 |
| Mất mát critic | 0,5 × MSE | Huber × 1 (bằng 0,5 × MSE khi sai số nhỏ) | 0,25 × MSE, có cắt | – |
| Cắt chuẩn gradient | 0,5 | 0,5 | 0,5 | – |
| Chuẩn hóa phần thưởng | không | chia 10 | – | có |

Đọc bảng:

- **Cấu hình của nhóm là cấu hình uvipen cộng ba thay đổi:** tốc độ học giảm dần, entropy giảm dần và `target_kl`.
  - uvipen báo vượt được 31/32 màn.
  - Họ chỉ chỉnh tốc độ học riêng cho từng màn (1e-3, 1e-4 hoặc 1e-5). Màn 1-3 được ghi là thành công ở 7e-5 sau 70 lần thất bại.
  - Họ không công bố số bước hay thời gian huấn luyện, nên không so được hiệu quả mẫu.
  - Tôi không tìm được nguồn nào công bố số bước PPO từ điểm ảnh cần để vượt 1-1 hay 4-1.
- **Hai chỗ nhóm khác hẳn phần còn lại của tài liệu là γ (0,9 so với 0,99–0,999) và số epoch (10 so với 3–4).** Đây cũng là hai chỗ số đo cho thấy chưa khớp (Mục 2 và 4).
- **Andrychowicz và cộng sự (2020) huấn luyện hơn 250.000 tác tử, nhưng chỉ trên 5 bài điều khiển liên tục MuJoCo, không có ảnh.** Vì vậy khuyến nghị của họ chỉ dùng để tham khảo:
  - γ là một trong những tham số quan trọng nhất, phải chỉnh theo môi trường, nên bắt đầu từ 0,99.
  - λ = 0,9.
  - Không cắt mất mát critic, vì cách đó làm kết quả kém đi.
  - Chuẩn hóa lợi thế ảnh hưởng ít.
  - Entropy không giúp rõ trên các bài của họ.
  - Tách mạng actor và critic tốt hơn dùng chung.

## 2. Chẩn đoán từ các lượt đã chạy

Các số dưới đây là trung vị hoặc trung bình trên mọi lần cập nhật của lượt. "Đầu → cuối" là trung bình của 1/5 đầu và 1/5 cuối lượt. Bảng đủ theo năm đoạn có trong đầu ra của `experiments/config_review.py`.

| Lượt | Lần cập nhật | Dừng sớm vì `target_kl` | Số epoch trung bình | approxKL (trung vị) | Tỉ lệ cắt, đầu → cuối | Entropy cuối | Explained var (trung vị) | Chuẩn gradient trước khi cắt (trung vị) |
|---|---|---|---|---|---|---|---|---|
| 1-1 (`run-1`) | 293 | 1,4% | 9,96 | 0,012 | 0,15 → 0,20 | 0,69 | 0,70 | 1,52 |
| 4-1 từ đầu | 293 | 0% | 10,00 | 0,010 | 0,19 → 0,09 | 0,54 | 0,74 | 1,45 |
| 1-1 + 4-1 | 733 | 35,5% | 9,05 | 0,019 | 0,25 → 0,09 | 0,31 | 0,74 | 1,09 |
| 2-3, 3 triệu bước | 733 | 31,1% | 8,94 | 0,019 | 0,26 → 0,10 | 0,38 | 0,70 | 1,17 |
| 8 màn, 10 triệu bước | 2.441 | 72,1% | 6,28 | 0,021 | 0,27 → 0,09 | 0,28 | 0,79 | 1,45 |
| 2-1, cắt 200 | 365 | 38,6% | 8,93 | 0,021 | 0,18 → 0,15 | 0,67 | 0,92 | 1,45 |
| 2-1, cắt 2000 | 365 | 19,7% | 9,41 | 0,020 | 0,22 → 0,19 | 1,04 | 0,94 | 1,27 |

- **approxKL.** Các lượt một màn ở khoảng 0,010–0,012. Như vậy bước cập nhật còn có thể dài hơn; ghi chú cũ trong `configs/group-4-1.yaml` cũng nói vậy. Các lượt nhiều màn ở khoảng 0,02, đúng ngưỡng mà bài "37 implementation details" coi là giới hạn nên giữ dưới. Ở đó `target_kl` 0,03 mới là thứ quyết định độ dài bước, không phải tốc độ học hay số epoch.
- **Tỉ lệ cắt 0,09–0,27** là hơi cao so với ε = 0,15. Đây là hệ quả bình thường của 10 epoch, không phải dấu hiệu hỏng.
- **Chuẩn gradient trước khi cắt có trung vị 1,1–1,5, trong khi trần là 0,5.**
  - Như vậy gần như mọi bước gradient đều bị cắt: ở cả bảy lượt, 100% số lần cập nhật có chuẩn trung bình lớn hơn 0,5.
  - Adam gần như không phụ thuộc vào độ lớn chung của gradient, nên phép cắt ở đây thực chất là chuẩn hóa gradient của từng mini-batch.
  - uvipen và CleanRL dùng cùng trần này.
  - Không có số đo nào cho thấy nó gây hại. Cần biết điều này khi đọc biểu đồ, và nó quan trọng khi tăng γ (Mục 4).
- **Explained variance 0,70–0,94.** Critic học tốt. Với λ = 1, mục tiêu của critic là lợi tức Monte Carlo có nhiễu, nên 0,7 đã là mức khá.

## 3. Đánh giá từng tham số

| Tham số | Giá trị | Bằng chứng | Kết luận |
|---|---|---|---|
| `num_envs` × `num_steps` | 8 × 512 | Giống uvipen. Một đoạn thu dài khoảng 2 lượt chơi (lượt dài 145–280 bước). Ở đoạn 0–2 triệu của lượt 8 màn, mỗi lần cập nhật chỉ có một quỹ đạo của mỗi màn. Burda và cộng sự (2018) thấy tăng số môi trường song song từ 128 lên 2048 tốt hơn nhiều, nhưng đó là bài học bằng tò mò (curiosity). | Giữ. (Gợi ý thử 16 × 256 ở bản đầu đã bị bác: đo trực tiếp không thấy ít nhiễu hơn, xem `docs/config-math.md`, Mục 2.) |
| `update_epochs`, `num_minibatches` | 10, 16 | Dừng sớm 0–1,4% số lần cập nhật khi học một màn, 31–72% khi học nhiều màn. CleanRL, SB3 Zoo, Procgen và PPG dùng 3–4 epoch; PPG thấy 3 epoch là tốt nhất cho PPO. | Không sai, vì `target_kl` tự cắt bớt. Khi học nhiều màn, đặt 4–6 epoch sẽ bớt 25–35% thời gian mỗi vòng (Mục 5), nhưng chưa biết có mất hiệu quả mẫu không. |
| `lr` | 2e-4, giảm về 0 | Nhóm đã thấy: các lượt chạy hết lịch kết thúc ổn định hơn (`nhat-ky-trong-tu.md`, Mục 14.2). Kinh nghiệm của uvipen: tốc độ học là tham số nhạy nhất theo màn. | Giữ. |
| `gamma` | 0,9 | Xem Mục 4. | Ứng viên số 1 cho một thí nghiệm. |
| `gae_lambda` | 1,0 | Với γ = 0,9 thì γλ = 0,9, nên lợi thế chỉ cộng dồn khoảng 10 bước và phương sai còn vừa phải. Các nguồn khác dùng 0,9–0,95. | Giữ nếu giữ γ. Nếu tăng γ thì hạ λ về khoảng 0,95, để γλ ≈ 0,92 gần như cũ. |
| `clip_coef` | 0,15 | Nằm giữa 0,1 (CleanRL) và 0,2 (uvipen, Procgen). | Giữ. |
| `ent_coef` → `ent_coef_final` | 0,01 → 0,002 | Đề chấm bằng cách lấy mẫu. Chọn hành động xác suất cao nhất tốt hơn lấy mẫu 5–20 điểm phần trăm ở mọi checkpoint đã so (ở 5-2 là 87% so với 73,5%). | Giữ. Với c = 0,002, phần thưởng entropy đã quá nhỏ để giữ độ ngẫu nhiên còn lại, nên đặt mức cuối bằng 0 sẽ gần như không đổi gì (`docs/config-math.md`, Mục 5). Không chỉnh logits sau khi huấn luyện. |
| `vf_coef` | 0,5 (trên MSE không có ½) | Bằng uvipen khi sai số nhỏ, gấp đôi CleanRL và gấp đôi mặc định 0,25 của thầy. Mạng dùng chung thân nên critic và actor tranh nhau gradient (PPG, bài 37 chi tiết). Explained variance vẫn tốt. | Giữ với γ = 0,9. Nếu tăng γ thì phải chuẩn hóa phần thưởng (Mục 4). |
| `max_grad_norm` | 0,5 | Luôn bị chạm trần (Mục 2). | Giữ, chỉ cần biết là nó luôn hoạt động. |
| `target_kl` | 0,03 | Là thứ quyết định độ dài bước khi học nhiều màn. | Giữ. |
| `adam_eps`, `norm_adv`, `clip_vloss` | 1e-5, có, không | Khớp bài 37 chi tiết. Andrychowicz thấy cắt mất mát critic làm kết quả kém đi. | Giữ. |
| Phần thưởng: `weights`, `flag_bonus`, `normalize_reward` | [1, 1, 2, 0, 0], 2,0, không | Khi chạy đều, mỗi bước được khoảng 0,33, nên V ≈ 3,3: đủ nhỏ để khỏi chuẩn hóa khi γ = 0,9. | Giữ. Khi tăng γ thì bật `normalize_reward`. |
| `noop_max`, `max_no_progress_steps` | 30, 200 | `noop_max` khớp giao thức chấm (0–30 bước NOOP). Thí nghiệm cắt lượt ở 2-1 cho 39 so với 10 lượt chạm cờ, nhưng mỗi bên chỉ một seed. | Giữ. |
| Mạng | Nature-CNN, 1,69 triệu tham số | Procgen: Nature-CNN "gần như không học được" ở đó; IMPALA-CNN tốt hơn cả hiệu quả mẫu lẫn khả năng chơi màn lạ. Nhưng Procgen sinh màn ngẫu nhiên, còn ở đây màn cố định. Đo trên máy này: IMPALA-CNN chậm hơn khoảng 10–12 lần mỗi bước gradient. | Giữ khi còn chạy trên CPU. |
| `allocation` (`floor` 0,06, `cap` 0,30) | Chia theo tỉ lệ hỏng lúc huấn luyện | 2-1 nhận 2,04 triệu bước (20%) và đạt 3,5%; 1-1, 4-1, 6-1 mỗi màn khoảng 0,95–0,98 triệu bước và đạt 97,5–99,5%. Mẹo cắt lượt làm sai số đo tỉ lệ hỏng của 2-1 (lúc huấn luyện ≤ 0,14 trong khi chấm được 42/100 ở bước 9,0 triệu). | Lượt sau nên đổi, xem Mục 7. |

Màn 2-1 cho thấy cách chia theo tỉ lệ hỏng sai ở đâu: tỉ lệ hỏng đo "màn còn khó", không đo "học thêm có tiến không". Prioritized Level Replay (Jiang và cộng sự, 2021) chấm mỗi màn bằng trung bình |GAE|, tức sai số của critic, và trên Procgen đưa điểm từ 100 lên 128,3 (so với chọn màn đều). Ở bức tường của 2-1, critic đã đoán đúng V ≈ 0 (`inside-the-network.md`, mục 1). Vì vậy |GAE| ở đó nhỏ, và một cách chấm như PLR sẽ thôi dồn bước vào 2-1. Điều này là suy luận, chưa đo.

## 4. Hệ số chiết khấu γ: phép tính

Một bước tác tử là 4 khung hình, tức 1/15 giây.

| γ | Tầm nhìn 1/(1−γ) | Nửa đời | Trọng số ở 10 / 30 / 60 bước | Khoảng cách xa nhất mà một sự kiện còn giữ ≥ 10% trọng số | V khi chạy đều |
|---|---|---|---|---|---|
| 0,9 | 10 bước (0,67 s) | 6,6 bước | 0,35 / 0,04 / 0,002 | 22 bước | 3,3 |
| 0,95 | 20 bước (1,33 s) | 13,5 bước | 0,60 / 0,21 / 0,05 | 45 bước | 6,7 |
| 0,97 | 33 bước (2,22 s) | 22,8 bước | 0,74 / 0,40 / 0,16 | 76 bước | 11,1 |
| 0,99 | 100 bước (6,67 s) | 69 bước | 0,90 / 0,74 / 0,55 | 229 bước | 33,3 |

**1. Critic "thấy trước ngắn" phần lớn là do γ.**
- `inside-the-network.md` đo được hai điều:
  - V của lượt sắp hỏng chỉ tụt rõ (ít nhất 0,5) từ 7–15 bước trước khi chết.
  - Kết cục của các lượt ấy đã xấu đi từ 23–26 bước trước.
- Xét một critic hoàn hảo với γ = 0,9. Chết sau k bước làm mất phần phạt 1,67 cộng phần thưởng tương lai khoảng 3,3, nên V giảm khoảng 5 × 0,9^k:
  - k = 15: giảm 1,03.
  - k = 20: giảm 0,61.
  - k = 25: giảm 0,36, đã dưới ngưỡng 0,5.
- Vậy ngay cả critic hoàn hảo cũng chỉ thấy tới khoảng 22 bước, ngắn hơn khoảng 23–26 bước mà kết cục đã định. Đây là giới hạn của γ, không phải điểm yếu của critic.

**2. Ở bức tường lò xo của 2-1, phần thưởng của việc thoát nhỏ so với cái giá của việc thử.**
- Lùi 1–2 bước tốn khoảng −0,2 tới −0,4 (`inside-the-network.md`, mục 4).
- Giá trị của "đã qua tường", nhìn từ D bước trước, là γ^D × V:
  - Với γ = 0,9: 1,97 / 1,16 / 0,69 / 0,41 khi D = 5 / 10 / 15 / 20.
  - Với γ = 0,97: 9,5 / 8,2 / 7,0 / 6,0.
- Nếu đường thoát dài 15–20 bước, phần thưởng gấp cái giá khoảng 1–3 lần với γ = 0,9, và khoảng 15–35 lần với γ = 0,97.
- Như vậy, với γ = 0,9, một hành động hiếm khi dẫn tới lối thoát dễ nhận lợi thế âm.
- Phép tính này khớp với cách hiểu mới nhất trong nhật ký (Mục 17): bấm "trái" bị trừ thưởng ngay, còn phần thưởng qua tường tới quá muộn. Nhưng nó dựa trên mô hình thô (tốc độ chạy đều, V = r/(1−γ)), nên chỉ nói γ là nghi phạm hợp lý. Nó không chứng minh γ là nguyên nhân. `inside-the-network.md`, mục 4, đã lập luận ngược lại, với giả định đường thoát chỉ dài 5–15 bước.

**3. Cái giá của γ lớn hơn.**
- V tăng từ 3,3 lên 11,1 (γ = 0,97) hoặc 33,3 (γ = 0,99). Mất mát critic vì thế lớn hơn khoảng 11 đến 100 lần.
- Chuẩn gradient luôn bị cắt về 0,5 (Mục 2). Khi critic lớn lên, gradient của nó sẽ chiếm gần hết phần 0,5 đó, và gradient của actor bị thu nhỏ theo. Nếu chỉ đổi γ, thí nghiệm sẽ thiên lệch, làm γ lớn trông kém hơn thật.
- Phần phạt khi chết và phần thưởng cờ cũng nhỏ đi so với V: từ 0,5 V còn 0,15 V hoặc 0,05 V.
- Cách bù đã có sẵn trong mã: `env.normalize_reward: true`. `make_env` dùng `NormalizeReward` với đúng γ của cấu hình, nên lợi tức về cùng cỡ ở mọi γ.

**Thí nghiệm đề xuất cho Mục 4 của báo cáo:**
- Đổi một thứ: γ = 0,9 so với 0,97.
- Bật `normalize_reward` ở **cả hai** nhánh, giữ nguyên mọi thứ khác, 2 seed mỗi nhánh.
- Đo:
  - tỉ lệ chạm cờ trên 100 lượt, seed 7;
  - số bước tới khi tỉ lệ chạm cờ lúc huấn luyện đạt 50%;
  - `value_loss` và `grad_norm`, để biết critic có lấn actor không.
- Nên chạy trên một màn học được trong khoảng 1,2 triệu bước (1-1 hoặc 4-1) để đo lợi hại chung. Nếu muốn đúng câu hỏi bức tường thì chạy thêm trên 2-1.
- Số bước do Dũng quyết. Một lượt 1-1 dài 1,2 triệu bước mất khoảng 2,5 giờ ở 140 bước mỗi giây, nên 4 lượt mất khoảng 10 giờ máy.

## 5. Thời gian chạy: chỗ tốn và GPU

Thời gian mỗi vòng được tách thành hai phần: thu dữ liệu, và các bước gradient. Phần thứ hai suy ra bằng hồi quy từ cột `sps`:

| Lượt | Thu dữ liệu và phần khác | Mỗi bước gradient | 10 epoch đủ | Phần của cập nhật trong mỗi vòng | Nếu 4 epoch |
|---|---|---|---|---|---|
| 8 màn | 17,5 s | 77 ms | 12,3 s | 41% | 22,4 s thay vì 29,8 s (−25%) |
| 1-1 + 4-1 | 14,3 s | 110 ms | 17,7 s | 55% | 21,3 s thay vì 31,9 s (−33%) |
| 2-3, 3 triệu bước | 12,9 s | 110 ms | 17,6 s | 58% | 19,9 s thay vì 30,5 s (−35%) |

Đo trực tiếp (`config_review.py --bench`, 6 luồng torch, mini-batch 256), mỗi bước gradient:
- Nature-CNN của nhóm: 9,3 triệu phép nhân-cộng mỗi ảnh, 76–97 ms.
- IMPALA-CNN 16-32-32: 54 triệu phép nhân-cộng mỗi ảnh, 924–939 ms.

Trên CPU, lượt 8 màn sẽ lâu hơn khoảng 4–5 lần nếu dùng IMPALA-CNN.

**GPU.**
- `nvidia-smi` trong WSL thấy GTX 1650 Ti (4 GB, driver 560.70), nhưng torch là bản `2.15.0+cpu`.
- Ước tính, chưa đo: phần cập nhật cần khoảng 1,1 nghìn tỉ phép nhân-cộng mỗi vòng, trên GPU có thể chỉ còn 1–2 giây. Khi đó mỗi vòng còn khoảng 15–19 giây thay vì 30, tức nhanh khoảng 1,6–2 lần. Phần còn lại bị giới hạn bởi trình giả lập.
- Muốn dùng GPU phải cài bản torch có CUDA, nên cài vào một venv riêng để không đụng môi trường đã sinh ra các bản nộp. Kết quả sẽ lệch chút ít so với CPU; điều này đề cho phép ("sai khác do ngẫu nhiên").

## 6. Rủi ro điểm của ba bản nộp

`evaluate.py` cố định mọi seed. Vì vậy chạy lại trên cùng máy với cùng phiên bản thư viện sẽ ra đúng 19/20, 20/20 và 18/20. Trên máy khác, sai số dấu phẩy động có thể làm khác các hành động được lấy mẫu. Khi đó nên coi lần chấm của thầy là một lần bốc mới:

| Bản nộp | Tỉ lệ thật ước tính | P(≥ 16/20) | Kỳ vọng phần điểm min(1, f/0,8) | Kỳ vọng mất |
|---|---|---|---|---|
| 1-1 (25 điểm) | 0,89 (89/100) | 0,939 | 0,995 | 0,12 điểm |
| 4-1 (15 điểm) | 0,96 (96/100) | 0,999 | 1,000 | 0 |
| 2-3 (10 điểm thưởng) | 0,865 (173/200) | 0,878 | 0,989 | 0,11 điểm |

Kỳ vọng mất tổng cộng khoảng 0,23 điểm, nên không đáng huấn luyện lại bản nào. Có thể thay bản 1-1 bằng mạng hai màn (99/100 seed 7, 20/20 seed 2026), nhưng cách này lợi nhiều nhất 0,12 điểm, và báo cáo phải giải thích thêm một cách huấn luyện khác.

## 7. Đề xuất, theo thứ tự ưu tiên

1. **Trước khi nộp: không đổi cấu hình và không đổi checkpoint.**
2. **Việc đáng làm nhất là báo cáo và trình bày (30 điểm).**
   - Bảng ở Mục 2 dùng được ngay cho Mục 3 của báo cáo (đường cong học).
   - Bảng ở Mục 1 dùng cho Mục 2 (thiết lập huấn luyện), kèm lý do của từng giá trị.
3. **Nếu còn thời gian máy, chạy một thí nghiệm đổi đúng một thứ, 2 seed.** Hai lựa chọn:
   - Thí nghiệm γ ở Mục 4.
   - Seed thứ hai cho thí nghiệm cắt lượt ở 2-1: 2 × 1,5 triệu bước, khoảng 5–6 giờ. Thí nghiệm này hiện có kết quả 39 so với 10 nhưng mỗi bên chỉ một seed.
4. **Nếu chạy lại nhiều màn:**
   - Thay cách chia theo tỉ lệ hỏng bằng điểm |GAE| như PLR, hoặc theo tốc độ tiến bộ (hướng 3 trong `extension-directions.md`). Hoặc ít nhất hạ `cap` xuống khoảng 0,2.
   - Đặt `update_epochs` 4–6.
   - (Hai gợi ý ở bản đầu, `ent_coef_final: 0` và 16 × 256, đã bị bác trong `docs/config-math.md`, Mục 2 và 5.)
5. **Chỉ khi đã bật GPU:**
   - Cài torch có CUDA trong venv riêng, nhanh khoảng 1,6–2 lần.
   - Khi đó mới đáng thử IMPALA-CNN, và nếu muốn mạng chơi được màn lạ thì thử thêm tăng cường ảnh (DrAC: cắt ngẫu nhiên tốt nhất ở 8–9 trên 16 trò Procgen).
6. **Không nên làm:**
   - Dùng IMPALA-CNN trên CPU.
   - Chỉnh logits hay nhiệt độ sau khi huấn luyện.
   - Đổi nhiều thứ cùng lúc rồi gọi là một thí nghiệm.

## 8. Giới hạn của bản rà soát

- Không chạy bước huấn luyện nào, nên mọi chỗ ghi "nên thử" là giả thuyết.
- Andrychowicz chỉ đo trên MuJoCo. Procgen và CoinRun sinh màn ngẫu nhiên, khác Mario với màn cố định. Không tìm được số bước công bố cho PPO học từ ảnh trên 1-1 hay 4-1 để so hiệu quả mẫu.
- Phép tính γ dùng mô hình thô:
  - chạy đều 10 điểm ảnh mỗi bước, nên mỗi bước được khoảng 0,33;
  - V = r/(1−γ);
  - cái giá của việc lùi lấy từ `inside-the-network.md`.
- Thời gian ở Mục 5 suy ra từ cột `sps`, vốn là trung bình cộng dồn, nên có nhiễu. Hai lần đo Nature-CNN lệch nhau 76 và 97 ms. Mức nhanh hơn khi dùng GPU là ước tính.
- Mọi lượt chạy chỉ có một seed.

## 9. Nguồn

- uvipen, Super-mario-bros-PPO-pytorch: [train.py](https://raw.githubusercontent.com/uvipen/Super-mario-bros-PPO-pytorch/master/train.py), [src/env.py](https://raw.githubusercontent.com/uvipen/Super-mario-bros-PPO-pytorch/master/src/env.py), [README](https://raw.githubusercontent.com/uvipen/Super-mario-bros-PPO-pytorch/master/README.md)
- CleanRL, [ppo_atari.py](https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_atari.py); Huang và cộng sự (2022), [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- RL Baselines3 Zoo, [hyperparams/ppo.yml](https://raw.githubusercontent.com/DLR-RM/rl-baselines3-zoo/master/hyperparams/ppo.yml)
- Andrychowicz và cộng sự (2020), What Matters in On-Policy RL, [arXiv 2006.05990](https://arxiv.org/abs/2006.05990)
- Cobbe và cộng sự (2020), Procgen, [arXiv 1912.01588](https://arxiv.org/abs/1912.01588); Cobbe và cộng sự (2019), CoinRun, [arXiv 1812.02341](https://arxiv.org/abs/1812.02341)
- Cobbe và cộng sự (2020), Phasic Policy Gradient, [arXiv 2009.04416](https://arxiv.org/abs/2009.04416)
- Jiang và cộng sự (2021), Prioritized Level Replay, [arXiv 2010.03934](https://arxiv.org/abs/2010.03934)
- Burda và cộng sự (2018), Large-Scale Study of Curiosity-Driven Learning, [arXiv 1808.04355](https://arxiv.org/abs/1808.04355)
- Raileanu và cộng sự (2021), DrAC, [arXiv 2006.12862](https://arxiv.org/abs/2006.12862)
