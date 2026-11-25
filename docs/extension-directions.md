# Ba hướng mở rộng sau lượt chạy 8 màn

Ghi ngày 24/11/2026, lúc lượt `runs/8-levels/from-scratch-10m` đang ở khoảng bước 8,7 triệu trên 10 triệu. Chưa hướng nào được bắt đầu.

## Vì sao cần mở rộng

- Ba checkpoint nộp (1-1, 4-1, 2-3) đã có. Lượt 8 màn nằm ngoài thang điểm; nó chỉ có giá trị qua báo cáo (25 điểm) và phần trình bày.
- Mục 4 của báo cáo đòi một thí nghiệm đổi đúng một thứ, có giả thuyết, kết quả, kết luận. Mọi so sánh đã có đều chỉ một seed hoặc khác nhau nhiều thứ cùng lúc.
- Lượt 8 màn chỉ có một seed và không có đối chứng, nên nó cho biết "được mấy màn", không cho biết vì sao.

Câu hỏi mà lượt 8 màn để lại: tính từ mốc 3,5 triệu bước, màn 3-1 đạt 81% với 630 nghìn bước riêng, còn màn 2-1 nhận 1,15 triệu bước mà ở 4% (số lúc huấn luyện, ở bước 8,66 triệu). Đã biết 2-1 vướng ở đâu, chưa biết vì sao mạng không học được.

Năm chỗ vướng của 2-1 ở checkpoint 6,3 triệu (100 lượt, seed 7):

| Vị trí x | Tình huống | Số lượt hỏng |
|---|---|---|
| 1000–1400 | đàn Goomba đi liền nhau và rùa | 15 |
| 1646 | nhảy lên ống đúng lúc cây ăn thịt trồi lên | 15 |
| 2066 | rơi vào khe giữa ống thấp và ống rất cao, đứng tới hết giờ | 10 |
| 2666 | rùa có cánh nhảy cùng các cặp Goomba | 18 |
| 3026 | tường gạch cao, phải dùng lò xo | 3 |

## Hướng 1: soi vào bên trong mạng (không huấn luyện thêm)

- **Câu hỏi.** Trước khi chết ở 2-1, mạng có "biết" mình sắp chết không? Nó nhìn vào đâu trên màn hình? Nó học kỹ năng hay chỉ thuộc 8 màn?
- **Cách đo.**
  - Vẽ giá trị V mà mạng dự đoán ở từng bước trước lúc chết, so với các lượt qua được cùng đoạn đó.
  - Che từng ô ảnh rồi xem xác suất hành động và V đổi thế nào.
  - Chấm mạng 8 màn trên các màn chưa từng học, so với mạng 2 màn (số đã có ở `runs/1-1+4-1/from-scratch/eval/all-levels/`).
- **Đọc kết quả.** V giảm sớm trước khi chết nghĩa là mạng thấy nguy hiểm nhưng hành động sai. V vẫn cao tới lúc chết nghĩa là mạng không nhận ra nguy hiểm. Kết quả này quyết định nên thử gì ở hướng 2.
- **Học được.** Phần critic làm gì; mạng dựa vào đâu để quyết định.
- **Chi phí.** Vài giờ, không tốn bước huấn luyện, làm được ngay khi lượt 8 màn xong.

### Dự đoán ghi trước khi có số đo (24/11/2026, 07:10)

Ba câu hỏi đặt ra trước khi phiên đo trả kết quả. Cột của Dũng để trống cho tới khi Dũng điền.

| Câu hỏi | Dự đoán của Claude | Dự đoán của Dũng | Số đo |
|---|---|---|---|
| Ở ống có cây ăn thịt (x 1646), V giảm trước khi Mario chết bao lâu? | Chỉ giảm trong 3–5 bước cuối; trước đó V gần như bằng các lượt đi qua được | | Không đo riêng được ở x 1646: checkpoint cuối chỉ còn 3 lượt chết ở đó. Ở cụm chết vì địch đông nhất (rùa có cánh, x 2658–2706, 11 lượt), V tụt rõ từ 7 bước trước; ở bước cuối V vẫn 2,43 so với mức thường 3,40. Dự đoán của Claude đúng hướng (rất ngắn), lệch về con số. |
| Khi kẹt trong khe ống (x 2066), mạng chọn hành động gì nhiều nhất? | Các hành động đi sang phải (phải, phải + chạy, phải + nhảy) chiếm trên 90%; "trái" dưới 2% | | Ở checkpoint cuối không còn lượt nào kẹt ở khe ống; 73/100 lượt kẹt ở tường lò xo x 3026. Ở đó bốn hành động có "phải" chiếm 57–81% tùy khoảng bước; đứng yên và nhảy tại chỗ chiếm phần còn lại; "trái" 0,00–0,01. Dự đoán của Claude sai ở phần "trên 90%", đúng ở phần "trái". |
| Trên các màn chưa học, mạng 8 màn đi xa hơn mạng 2 màn bao nhiêu (x trung bình)? | Khoảng 1,3–1,6 lần; không chạm cờ ở màn chưa học nào | | 378 so với 332 điểm ảnh trên 24 màn, tức 1,14 lần; 0/480 lượt chạm cờ. Mạng 8 màn còn kém ba mạng một màn (407–453). Dự đoán của Claude quá cao ở tỉ số, đúng ở phần không chạm cờ. |

Số đo lấy từ `docs/inside-the-network.md` (24/11/2026, checkpoint cuối 9.998.336, 100 lượt seed 7).

### Điều hướng 1 làm thay đổi

- Bảng "năm chỗ vướng" ở đầu tệp là của checkpoint 6,3 triệu. Ở checkpoint cuối, 2-1 kết thúc như sau: hết giờ 73, địch 19, rơi hố 6, chạm cờ 2. Cả 73 lượt hết giờ đều đứng trước bức tường lò xo ở x 3026; khe ống x 2066 không còn lượt nào kẹt.
- Ở bức tường, critic hiểu tình huống (V ≈ 0 khi kẹt, tăng lên 1,1–1,6 khi Mario đứng lên lò xo), nhưng chính sách gần như không bấm "trái" nên không lùi lên lò xo.
- Ở checkpoint 6,3 triệu, 8 trên 11 lượt tới tường còn thoát được, và 6 trong 8 lần thoát đó cần hơn 200 bước đứng yên. Lúc huấn luyện, lượt bị cắt sau đúng 200 bước không tiến. Tới checkpoint cuối, số lần thoát trên 1000 bước kẹt giảm từ 1,319 xuống 0,016.
- Vì vậy biến được chọn cho hướng 2 là ngưỡng cắt lượt, không phải γ (xem mục "Thí nghiệm đang chạy" ở hướng 2).

## Hướng 2: thí nghiệm có đối chứng ở màn 2-1, hệ số chiết khấu γ

- **Giả thuyết.** γ = 0,9 nhìn quá ngắn: phần thưởng cách 30 bước chỉ còn giá trị 4% (0,9³⁰). Vì vậy Mario không học được việc lùi lại để thoát khe ống (x 2066) hay lấy đà ở tường lò xo (x 3026).
- **Dự đoán kiểm được.** γ = 0,97 giúp hai đoạn đó, nhưng không giúp đoạn rùa có cánh (x 2666), vì đó là phản xạ.
- **Cách đo.** Hai giá trị γ, giữ nguyên mọi thứ khác, 2 seed mỗi bên, huấn luyện chỉ trên 2-1. So tỉ lệ qua từng đoạn bằng `experiments/death_look.py`, không chỉ tỉ lệ chạm cờ.
- **Giả thuyết phụ chưa kiểm.** Mẹo cắt lượt khi Mario không tiến (`max_no_progress_steps`) có thể đã lấy mất chính những bước cần để học cách thoát khe ống.
- **Học được.** Gán công trạng theo thời gian (credit assignment). Sai dự đoán cũng là một kết quả viết được vào Mục 4 của báo cáo.
- **Chi phí.** 4 lượt nhỏ, khoảng một đêm máy. Số bước do Dũng quyết.

### Thí nghiệm cắt lượt ở 2-1 (chạy 24/11/2026, 08:08–13:46)

Dũng duyệt ngày 24/11/2026: tối đa 3 triệu bước huấn luyện, hai lượt chỉ trên 2-1. Biến do Claude chọn sau khi có kết quả hướng 1; thiết kế dưới đây cũng do Claude chọn lúc Dũng vắng.

- **Giả thuyết.** Mẹo cắt lượt sau 200 bước không tiến đã cắt mất những lần thoát khỏi tường lò xo, nên mạng mất dần cách thoát.
- **Thiết kế.** Hai nhánh cùng nối tiếp checkpoint 6.303.744 của lượt 8 màn (lúc mạng còn thoát được), cùng học chỉ màn 2-1 thêm 1.495.040 bước, cùng lịch tốc độ học và lịch entropy của lượt gốc. Khác nhau đúng một khóa `env.max_no_progress_steps`:
  - nhánh thử `runs/2-1/wall-cut-2000`: 2000 (gần như tắt cắt lượt);
  - nhánh đối chứng `runs/2-1/wall-cut-200`: 200 (như lượt gốc).
- **Cấu hình.** `configs/group-2-1-experiment.yaml`. Không đụng thư mục lượt 8 màn, chỉ đọc checkpoint của nó.
- **Dự đoán kiểm được.** Ở nhánh 2000, số lần thoát trên 1000 bước kẹt ở tường không sụp về gần 0 và xác suất "trái" khi kẹt giữ trên khoảng 0,05; ở nhánh 200 thì sụp như lượt gốc. Đo bằng `experiments/value_trace.py` trên checkpoint cuối mỗi nhánh (100 lượt, seed 7).
- **Giới hạn biết trước.** Mỗi nhánh một seed. Hai nhánh chạy lần lượt để máy không quá tải. Tổng số bước: 2.990.080 cho hai nhánh, cộng 8.192 bước chạy thử (`runs/2-1/wall-smoke`).
- **Vì sao không học từ đầu.** Học 2-1 từ đầu với 1,5 triệu bước có thể chưa tới được bức tường, khi đó hai nhánh không có gì để so. Nối tiếp từ 6,3 triệu thì hiện tượng cần kiểm (mất cách thoát) đã quan sát được ngay từ điểm xuất phát đó.

### Kết quả nhánh thử (cắt 2000 bước), 24/11/2026, 11:20

Phần này viết khi nhánh đối chứng chưa xong; kết luận của thí nghiệm ở mục kế tiếp. "Ở tường" là các bước có x trong 2960–3060; "qua tường" là lượt có x lớn nhất trên 3130. Số đo từ `trace.npz` (100 lượt, seed 7, không cắt lượt).

| Checkpoint | Chạm cờ | Lượt tới tường | Lượt qua tường | Lượt qua trên 1000 bước ở tường | Xác suất "trái" khi đứng ở tường |
|---|---|---|---|---|---|
| Điểm xuất phát, lượt 8 màn, bước 6.303.744 | 8 | 11 | 8 | 1,29 | 0,085 |
| Nhánh thử, cắt 2000, bước 6.803.456 | 1 | 14 | 1 | 0,05 | 0,010 |
| Nhánh thử, cắt 2000, bước 7.303.168 | 2 | 8 | 2 | 0,17 | 0,009 |
| **Nhánh thử, cắt 2000, bước 7.798.784 (cuối nhánh)** | 10 | 44 | 10 | 0,16 | 0,015 |
| Lượt 8 màn, bước 9.003.008 (để tham khảo) | 42 | 43 | 42 | 1,57 | 0,079 |
| Lượt 8 màn, bước cuối (để tham khảo) | 2 | 75 | 2 | 0,016 | 0,004 |

- **Dự đoán cho nhánh thử sai.** Tôi đoán tỉ lệ qua tường không sụp và xác suất "trái" giữ trên 0,05. Thực tế tỉ lệ qua tường giảm 8 lần (1,29 xuống 0,16) và xác suất "trái" còn 0,015, dù lượt gần như không bị cắt.
- **Bỏ ngưỡng cắt không đủ để mạng học cách qua tường.** Sau 1,5 triệu bước chỉ học 2-1, số chạm cờ khi chấm là 10/100, so với 8/100 ở điểm xuất phát. Tỉ lệ chạm cờ lúc huấn luyện cao nhất 0,20 (bước 7.643.136), cuối nhánh 0,09.
- **Mạng tới tường nhiều hơn (44 lượt so với 11) nhưng qua tường kém hơn.** Kiểu thay đổi này giống lượt 8 màn ở cuối lượt, chỉ nhẹ hơn.
- **Khả năng qua tường mất ngay trong nửa triệu bước đầu của nhánh.** Ở bước 6.803.456 xác suất "trái" khi đứng ở tường đã còn 0,010 (từ 0,085) và chỉ 1 trên 14 lượt tới tường qua được. Phần đường trước tường thì chưa khá lên ở hai checkpoint giữa: số lượt chết vì địch là 70 và 67 trên 100, bằng điểm xuất phát (67), và chỉ giảm còn 46 ở cuối nhánh. Các tỉ lệ ở hai checkpoint giữa dựa trên 1 và 2 lần qua tường, nên chỉ đọc được bậc độ lớn.
- Tiêu chí ghi trước khi có nhánh đối chứng: nếu nhánh cắt 200 cho tỉ lệ qua tường thấp hơn hẳn 0,16 thì ngưỡng cắt có góp phần; nếu hai nhánh ngang nhau thì giả thuyết sai.

### Kết quả cả hai nhánh, 24/11/2026, 14:10

Nhánh đối chứng (cắt 200) xong lúc 13:46. Tổng số bước huấn luyện đã dùng: 2.998.272 (hai nhánh 2.990.080, chạy thử 8.192). Mỗi checkpoint được đo 100 lượt, seed 7, không cắt lượt, bằng `experiments/value_trace.py`; tệp ở `runs/2-1/wall-cut-2000/eval/` và `runs/2-1/wall-cut-200/eval/`.

| Checkpoint | Chạm cờ | Lượt tới tường | Lượt qua tường | Qua sau ≤ 200 bước ở tường | Lượt qua trên 1000 bước ở tường | Xác suất "trái" khi đứng ở tường | Chết vì địch |
|---|---|---|---|---|---|---|---|
| Điểm xuất phát, bước 6.303.744 | 8 | 11 | 8 | 2 | 1,29 | 0,085 | 67 |
| Cắt 2000, bước 6.803.456 | 1 | 14 | 1 | 0 | 0,05 | 0,010 | 70 |
| Cắt 200, bước 6.803.456 | 1 | 16 | 1 | 1 | 0,04 | 0,027 | 62 |
| Cắt 2000, bước 7.303.168 | 2 | 8 | 2 | 0 | 0,17 | 0,009 | 67 |
| Cắt 200, bước 7.303.168 | 12 | 31 | 12 | 0 | 0,28 | 0,044 | 56 |
| **Cắt 2000, bước 7.798.784 (cuối)** | 10 | 44 | 10 | 3 | 0,16 | 0,015 | 46 |
| **Cắt 200, bước 7.798.784 (cuối)** | 39 | 60 | 39 | 0 | 0,44 | 0,017 | 34 |

Tỉ lệ chạm cờ lúc huấn luyện: nhánh cắt 2000 cao nhất 0,20 (bước 7.643.136), cuối nhánh 0,09; nhánh cắt 200 cao nhất 0,08 (bước 7.135.232), cuối nhánh 0,00.

Tỉ lệ qua tường theo thời gian đã đứng ở tường, ba checkpoint (số bước ở tường / số lần qua / xác suất "trái" trung bình ở các bước đó):

| Đã đứng ở tường | Đồng hồ trò chơi (trung bình) | Cắt 200, bước 7.798.784 | Cắt 2000, bước 7.798.784 | Lượt 8 màn, bước 9.003.008 |
|---|---|---|---|---|
| 0–200 bước | 321 | 12.000 / 0 / 0,002 | 8.536 / 3 / 0,018 | 8.254 / 4 / 0,016 |
| 200–500 bước | 271 | 18.000 / 0 / 0,003 | 12.148 / 2 / 0,018 | 10.050 / 18 / 0,069 |
| 500–850 bước | 207 | 21.000 / 0 / 0,005 | 12.711 / 4 / 0,021 | 5.025 / 11 / 0,134 |
| 850–1300 bước | 128 | 23.666 / 20 / 0,024 | 15.349 / 1 / 0,011 | 2.794 / 7 / 0,171 |
| trên 1300 bước | 43 | 13.132 / 19 / 0,059 | 13.804 / 0 / 0,008 | 645 / 2 / 0,176 |

- **Giả thuyết không được ủng hộ.** Nhánh bỏ cắt lượt kết thúc kém hơn nhánh giữ cắt 200: 10/100 so với 39/100 lượt chạm cờ, 0,16 so với 0,44 lần qua trên 1000 bước ở tường. Nếu coi các lần qua tường là độc lập thì chênh lệch này khó là ngẫu nhiên (kiểm định nhị thức chính xác, p = 0,002). Theo tiêu chí ghi trước thì giả thuyết sai: bỏ ngưỡng cắt không giữ được cách qua tường.
- **Dự đoán sai ở cả hai nhánh.** Tôi đoán nhánh 2000 giữ được tỉ lệ qua tường còn nhánh 200 sụp như lượt gốc; thực tế ngược lại.
- **Các lần qua tường của nhánh cắt 200 nằm ngoài những gì nó gặp lúc huấn luyện.** Cả 39 lượt qua tường đều đã đứng ở tường từ 852 bước trở lên (trung vị 1267); lúc huấn luyện, lượt như vậy bị cắt sau 200 bước, nên tỉ lệ chạm cờ lúc huấn luyện của nhánh này là 0,00. Trong 850 bước đầu ở tường nó không qua lần nào (0 trên 51.000 bước) và gần như không bấm "trái" (0,002–0,005); sau đó xác suất "trái" tăng lên 0,024 rồi 0,059 và mọi lần qua tường xảy ra ở đây. Lượt 8 màn ở bước 9.003.008 có cùng kiểu: xác suất "trái" tăng từ 0,016 lên 0,176 theo thời gian đứng, tỉ lệ qua tường tăng từ 0,48 lên 3,10 trên 1000 bước.
- **Nhánh cắt 2000 thì gặp các trạng thái đó lúc huấn luyện, và ở đó nó bấm "trái" ít đi chứ không nhiều lên** (0,021 xuống 0,008). Nó qua tường ít hơn, nhưng 3 trên 10 lần qua xảy ra trong 200 bước đầu ở tường, so với 0 trên 39 của nhánh cắt 200.
- **Không nhánh nào học được cách qua tường ngay khi tới nơi.** Số lượt qua tường trong 200 bước đầu: 3 trên 44 lượt tới tường (cắt 2000), 0 trên 60 (cắt 200), so với 2 trên 11 ở điểm xuất phát.
- **Cách hiểu lúc đó (đã kiểm ở mục kế tiếp và sai):** đứng lâu thì đồng hồ trên màn hình xuống thấp; mạng học với cắt 200 chưa từng thấy bức tường với đồng hồ thấp, nên ở đó nó hành xử khác (bấm "trái" nhiều hơn) và tình cờ lùi được lên lò xo. Nếu đúng, số chạm cờ khi chấm ở 2-1 của các mạng học với cắt 200 (kể cả 42/100 ở bước 9,0 triệu) không đo một kỹ năng đã học, và điều đó giải thích được vì sao nó dao động mạnh giữa các checkpoint (8, 42, 5, 2). Số liệu ở đây không tách được "đồng hồ thấp" khỏi "đã đứng lâu"; muốn kiểm phải sửa riêng vùng đồng hồ trên khung hình rồi đo lại xác suất "trái".
- **Giới hạn.** Mỗi nhánh một seed huấn luyện, hai nhánh chạy lần lượt. Trong một lượt chạy, số chạm cờ ở 2-1 từng đổi từ 42 xuống 5 trong 0,9 triệu bước, nên chênh lệch 39 so với 10 giữa hai checkpoint đơn lẻ chưa đủ để nói cắt 200 tốt hơn cắt 2000. Ở mốc 6,8 triệu hai nhánh ngang nhau (1 và 1); chênh lệch chỉ có ở hai mốc sau.

### Kiểm cách hiểu về đồng hồ, 24/11/2026, 15:10 (chỉ đo, không huấn luyện)

Dũng duyệt phép kiểm này. Script: `experiments/clock_swap.py` (dán vùng đồng hồ, chấm hành vi) và `experiments/wall_scroll.py` (dán từng vùng khung hình, đo vị trí bức tường trên màn hình). Tệp ở `runs/2-1/wall-cut-200/eval/clock/`, `runs/2-1/wall-cut-2000/eval/clock/` và `runs/8-levels/from-scratch-10m/eval/clock-2-1-step9003008/`.

**Cách hiểu "mạng đọc đồng hồ" sai.**

- **Dán đồng hồ khác vào không đổi gì.** Lấy cảnh Mario đứng ở tường theo năm nhóm thời gian đã đứng, dán vào đó vùng đồng hồ của một nhóm khác: xác suất bấm "trái" lệch nhiều nhất 0,004 so với khi không sửa, ở cả 25 ô, trên cả ba checkpoint. Ví dụ ở nhánh cắt 200: cảnh đã đứng trên 1300 bước cho 0,063 khi không sửa và 0,062 khi dán đồng hồ lúc mới tới; cảnh mới tới cho 0,002 dù dán đồng hồ nào.
- **Chấm hành vi cũng không đổi.** Nhánh cắt 200, 100 lượt seed 7: không sửa 39 lượt chạm cờ (39 trên 60 lượt tới tường qua được, 0,44 lần qua trên 1000 bước ở tường); giữ đồng hồ đứng yên từ lúc tới tường 37 lượt chạm cờ (38 trên 55, 0,49); cho thấy đồng hồ còn 40 ngay khi tới tường 34 lượt chạm cờ (34 trên 56, 0,42). Ở cả hai kiểu sửa không lượt nào qua tường trong 200 bước đầu.
- **Cả dải HUD không có tác dụng; phần sân chơi thì có.** Ở nhánh cắt 200, dán toàn bộ sân chơi (hàng 11–83) của cảnh mới tới vào cảnh đứng lâu thì xác suất "trái" từ 0,047 xuống 0,003; dán ngược lại thì từ 0,003 lên 0,044. Dán dải HUD thì không đổi (0,046 và 0,003).

**Thứ thay đổi theo thời gian là màn hình bị cuộn.** Khi Mario ép vào tường, trò chơi vẫn cuộn màn hình dần sang phải, nên bức tường trôi sang trái trên khung hình và dừng khi mép trái của nó tới cột 30–31 (trên 84). Ở nhánh cắt 200, mép đó ở cột 41–44 trong 200 bước đầu và trung bình ở cột 34,6 sau 1300 bước; ở hai checkpoint kia màn hình cuộn nhanh hơn (trung bình cột 31 sau 850 bước). Mario vẫn ở đúng x 3026 của màn chơi.

Xác suất bấm "trái" theo cột của bức tường (Mario ở x 3026, y 79; trong ngoặc là số quan sát):

| Cột của bức tường | Cắt 200, bước 7.798.784 | Cắt 2000, bước 7.798.784 | Lượt 8 màn, bước 9.003.008 |
|---|---|---|---|
| 31 (đã cuộn hết) | 0,108 (149) | 0,009 (1037) | 0,176 (196) |
| 33 | 0,038 (61) | 0,006 (62) | 0,124 (26) |
| 35 | 0,011 (244) | 0,069 (29) | 0,034 (34) |
| 37 | 0,011 (58) | 0,033 (69) | 0,005 (83) |
| 39 | 0,005 (83) | – | 0,002 (91) |
| 41 | 0,002 (474) | – | 0,033 (70) |
| 43 (lúc mới tới) | 0,002 (123) | 0,008 (141) | 0,005 (21) |

- **Ở cùng một cột, xác suất "trái" không đổi theo thời gian đã đứng.** Ví dụ nhánh cắt 200, cột 35: 0,011 ở cả bốn nhóm thời gian có số liệu; cột 41: 0,002–0,003 ở cả năm nhóm. Vậy thời gian chỉ tác động qua việc màn hình cuộn.
- **Mỗi nhánh bấm "trái" rất ít ở những cảnh nó gặp nhiều lúc huấn luyện.** Nhánh cắt 200 bị cắt trước khi màn hình kịp cuộn (trong 200 bước đầu bức tường ở cột 41–44), và ở các cột đó xác suất "trái" của nó là 0,002; ở cột 31, cảnh nó hầu như không gặp lúc huấn luyện, là 0,108. Nhánh cắt 2000 đứng ở cột 31 phần lớn thời gian (1037 trên 1702 quan sát), và ở đó xác suất "trái" của nó là 0,009.
- **Cách hiểu mới, chưa kiểm:** ở cảnh nào được luyện thì PPO đẩy xác suất "trái" xuống, vì bấm "trái" bị trừ thưởng ngay (thưởng tính theo x) còn phần thưởng của việc lên lò xo rồi qua tường tới sau hàng chục bước, với γ = 0,9 thì gần như không còn giá trị. Các lần qua tường khi chấm đến từ những cảnh việc huấn luyện chưa chạm tới. Điều này đưa câu hỏi về lại giả thuyết γ ở đầu Hướng 2; kiểm nó cần huấn luyện (hai nhánh γ = 0,9 và γ = 0,97 nối từ cùng một checkpoint), số bước do Dũng quyết.
- **Giới hạn.** Số đo là tương quan giữa vị trí bức tường và xác suất "trái", cộng phép dán vùng sân chơi; chưa có phép dán riêng "bức tường dịch đi mấy cột" trong khi giữ nguyên mọi thứ khác. Nhận xét "nhánh cắt 200 hầu như không gặp cột 31 lúc huấn luyện" suy từ hành vi lúc chấm, không đo trên dữ liệu huấn luyện. Mỗi nhánh một seed.

## Hướng 3: công thức chia bước tự thiết kế (lớn nhất, nếu còn thời gian)

- **Vấn đề.** Công thức hiện tại chia bước theo tỉ lệ hỏng, nên đã dồn 29% số bước vào 2-1 mà không thu lại gì.
- **Hướng sửa.** Chia theo tốc độ tiến bộ thay vì theo tỉ lệ hỏng.
- **Tài liệu liên quan.** Teacher–Student Curriculum Learning (Matiisen và cộng sự, 2017); Prioritized Level Replay (Jiang và cộng sự, 2021).
- **Học được.** Curriculum tự động: chọn bài tập nào cho mạng học tiếp.
- **Chi phí.** Phải chạy nhiều màn nên tốn nhất; chỉ nên làm bản thu nhỏ khoảng 3 màn.

## Cách làm

Trước mỗi thí nghiệm, Dũng viết dự đoán bằng con số cụ thể rồi mới chạy, sau đó so lại. Dũng có thể tự viết một phần mã (ví dụ script vẽ V) và nhờ soát.

Thứ tự đề nghị: hướng 1 trước, vì nó không tốn bước và kết quả của nó định hình hướng 2.
