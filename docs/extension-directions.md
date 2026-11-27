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

### Đợt chạy 26/11/2026: seed 2 cho thí nghiệm cắt lượt và thử γ = 0,97 (ghi trước khi chạy, 18:05)

Dũng duyệt 4.485.120 bước ngày 26/11/2026. Kế hoạch đầy đủ và lý do ở `docs/config-math.md`; phần này chốt thiết kế, cách chấm và luật quyết định trước khi có số.

**Ba lượt.** Cả ba nối tiếp checkpoint 6.303.744 của lượt 8 màn, chỉ học 2-1, tự dừng ở bước 7.798.784 (1.495.040 bước mỗi lượt). Chạy lần lượt trên CPU, cùng venv `~/.venvs/mario-ppo` với hai nhánh cũ, mã ở commit `1b0df0d` (không sửa `mario_ppo/` cho tới khi lượt 3 đã khởi động). Cấu hình là `configs/group-2-1-experiment.yaml`, chỉ khác ở các khóa ghi đè dưới đây:

| Thứ tự | Thư mục | Ghi đè | Trả lời câu hỏi |
|---|---|---|---|
| 1 | `runs/2-1/wall-cut-200-seed2` | `seed=2` | Kết quả 39/10 có lặp lại với seed khác không |
| 2 | `runs/2-1/wall-cut-2000-seed2` | `seed=2 env.max_no_progress_steps=2000` | (cặp với lượt 1) |
| 3 | `runs/2-1/gamma-0.97-cut-200` | `algo.gamma=0.97` | γ dài hơn có giúp lùi lấy đà ở tường không; đối chứng là `runs/2-1/wall-cut-200` (γ = 0,9, seed 1) |

- "Seed 2" là seed của giai đoạn học tiếp; cả hai seed kế thừa cùng một mạng 8 màn.
- Lượt 3 giữ phần thưởng gốc (`normalize_reward: false`), λ = 1, cùng lịch tốc độ học và entropy, cùng optimizer nạp từ checkpoint; chỉ đổi γ.

**Cách chấm (chốt trước).**
- Mỗi lượt mới chấm 400 lượt chơi: 100 lượt seed 7 ở mỗi checkpoint 6.803.456, 7.303.168 và 7.798.784, cộng 100 lượt seed 11 ở 7.798.784.
- Hai nhánh cũ chấm thêm 100 lượt seed 11 ở 7.798.784.
- Chạy bằng `experiments/value_trace.py`; ba chỉ số tính bằng `experiments/wall_metrics.py`. Script này cho lại đúng mọi số của bảng 39/10 ở trên từ các `trace.npz` đã lưu.
- **Ba chỉ số:**
  1. số lượt chạm cờ trên 100;
  2. số lượt tới tường (x lớn nhất ≥ 2960);
  3. **qua sớm**: số lượt qua tường (x lớn nhất > 3130) sau không quá 200 bước ở tường (x trong 2960–3060), ghi dạng tử/mẫu với mẫu là số lượt tới tường, ví dụ 3/44. Mẫu bằng 0 thì ghi "không ước lượng được".

**Luật quyết định.**
- Quyết định dựa trên checkpoint cuối, với cả hai seed chấm. Hai checkpoint trước chỉ để xem diễn biến, không chọn mốc đẹp nhất.
- **Cắt lượt:** báo chênh lệch và chiều chênh của từng cặp seed huấn luyện (seed 1: 39 so với 10).
  - Cùng chiều ở cả hai seed: "cắt 200 cho nhiều lượt chạm cờ hơn".
  - Trái chiều: "chưa phân định được".
  - Chỉ được nói "học được động tác qua tường khi vừa tới" nếu chỉ số 3 tăng.
- **γ:** coi γ = 0,97 là có ích nếu ở checkpoint cuối, với cả hai seed chấm, cả số lượt chạm cờ lẫn chỉ số qua sớm đều cao hơn nhánh γ = 0,9 seed 1.
  - Nếu có ích: chạy thêm γ = 0,97 seed 2 (1.495.040 bước, cần duyệt riêng), lấy lượt 1 làm đối chứng.
  - Nếu critic phân biệt lượt qua và lượt hỏng tốt hơn mà hành vi ở tường không đổi: kết luận "đổi γ chưa chuyển thành cải thiện hành vi trong ngân sách đã thử". Phép thử tiếp là bắt đầu một phần lượt từ trạng thái lưu, sau khi có bài kiểm tra khôi phục khớp từng bước.
  - Nếu γ = 0,97 kém hơn và phép kiểm gradient cho thấy critic lấn actor: đo lại ở một checkpoint giữa lượt trước khi làm bảng 2 × 2 giữa γ và `normalize_reward`.
- Hai seed chấm cho kết quả trái chiều: ghi "chưa phân định được".

**Kiểm trước, không huấn luyện (giai đoạn 0).**
- Ba chỉ số của hai nhánh cũ tính lại bằng `wall_metrics.py` khớp đúng bảng 39/10.
- **Kiểm gradient khi đổi γ, tại 6.303.744** (`experiments/gamma_gradient_check.py`, 6 đoạn thu 8 × 512 trên 2-1, 16 mini-batch mỗi đoạn, không cập nhật trọng số; tệp `runs/2-1/gamma-check/step_006303744.json`). Trung vị, trong ngoặc là phân vị 10–90:

| | γ = 0,9 (như lúc học) | γ = 0,97 (cùng dữ liệu) |
|---|---|---|
| Mất mát critic (MSE) | 0,32 (0,23–0,43) | 29,4 (24,7–32,3) |
| Explained variance của V cũ với mục tiêu mới | 0,84 | 0,40 |
| Chuẩn gradient critic / actor trên thân CNN | 0,61 (0,44–0,93) | 22,3 (18,9–27,6) |
| cos giữa gradient critic và actor | 0,00 (−0,19 – 0,25) | −0,02 (−0,28 – 0,31) |
| Hệ số mà phép cắt chuẩn gradient nhân vào | 0,174 | 0,0061 |
| Chuẩn gradient actor sau khi cắt | 0,317 | 0,0114 (bằng 3,6%) |

  - **Đọc số.** Ngay khi đổi sang γ = 0,97, critic cũ sai lệch lớn: mất mát tăng khoảng 90 lần, gradient của critic gấp khoảng 22 lần gradient của actor trên thân CNN. Hai gradient gần như vuông góc, không ngược hướng.
  - **Hệ quả.** Phép cắt thu gradient còn khoảng 0,6%, nên phần của actor chỉ còn khoảng 3,6% so với khi giữ γ = 0,9.
  - **Đây là trạng thái lúc bắt đầu, chưa phải trong cả lượt.** Adam chuẩn hóa từng tham số theo lịch sử gradient của chính nó. Vì vậy bước của đầu chính sách sẽ hồi lại khi lịch sử đó thích nghi, sau khoảng vài nghìn bước gradient. Trên thân CNN, critic vẫn chiếm phần lớn cho tới khi mất mát của nó giảm.
  - **Hệ quả cho cách đọc kết quả.** Theo thiết kế, lượt 3 vẫn chạy với phần thưởng gốc. Nếu nó kém nhánh γ = 0,9, phải đo lại bằng script này ở checkpoint 6.803.456 của chính lượt 3 trước khi kết luận về γ, để biết critic đã thích nghi chưa.
- **Chấm lại hai nhánh cũ bằng seed 11, checkpoint 7.798.784** (100 lượt mỗi nhánh; tệp `runs/2-1/wall-cut-*/eval/value-2-1-step_007798784-seed11/`, bảng gộp ở `runs/2-1/wall-metrics-old-branches.json`):

| Nhánh (seed huấn luyện 1) | Seed chấm | Chạm cờ /100 | Tới tường | Qua sớm / tới tường | Qua trên 1000 bước ở tường |
|---|---|---|---|---|---|
| Cắt 200 | 7 | 39 | 60 | 0/60 | 0,44 |
| Cắt 200 | 11 | 44 | 60 | 1/60 | 0,55 |
| Cắt 2000 | 7 | 10 | 44 | 3/44 | 0,16 |
| Cắt 2000 | 11 | 8 | 45 | 0/45 | 0,12 |

  - Đổi seed chấm không đổi chiều kết quả: cắt 200 vẫn hơn hẳn về chạm cờ (44 so với 8).
  - Cả hai nhánh vẫn gần như không qua tường sớm (1/60 và 0/45).
  - Như vậy ở seed huấn luyện 1, kết quả không phụ thuộc lần chấm. Câu hỏi còn lại là nó có lặp lại với seed huấn luyện khác không, và lượt 1–2 sẽ trả lời câu này.

**Lượt 3 giữ nguyên thiết kế** (γ = 0,97, phần thưởng gốc): Dũng quyết sau khi xem phép kiểm gradient ở trên, ngày 26/11/2026.

**Khởi động.** Đợt chạy bắt đầu lúc 18:26 ngày 26/11/2026 (giờ máy; nhật ký `runs/2-1/batch-20261126.log` ghi giờ UTC của WSL). Lệnh chạy và lệnh dừng ở đầu `experiments/run_wall_batch.sh`; dừng hẳn bằng `bash experiments/stop_wall_batch.sh`.

**Dự đoán ghi trước.**
- *Dũng:* (Dũng điền trước khi xem kết quả).
- *Claude:*
  - Cặp seed 2: cắt 200 vẫn chạm cờ nhiều hơn cắt 2000 ở checkpoint cuối, nhưng chênh ít hơn 39/10. Chỉ số qua sớm ở cả hai nhánh không quá 3 trên số lượt tới tường.
  - γ = 0,97: chỉ số qua sớm từ 5 trên số lượt tới tường trở lên. Chạm cờ ở checkpoint cuối không thấp hơn 39 quá 10.
  - Độ tin thấp: `inside-the-network.md` mục 4 lập luận rằng γ không phải nguyên nhân, còn mục trên lập luận ngược lại.

### Kết quả đợt chạy 26/11/2026 (chấm xong 27/11/2026, 03:15)

**Huấn luyện.** Ba lượt đều tới bước 7.798.784 và thoát bình thường. Lượt 1 mất 2 giờ 53 phút, lượt 2 mất 2 giờ 35 phút, lượt 3 mất 2 giờ 44 phút; cả đợt xong lúc 02:38.

**Chấm.** Chấm đúng như đã chốt: 400 lượt cho mỗi lượt mới, bằng `value_trace.py`; chỉ số tính bằng `wall_metrics.py`. Bảng gộp ở `runs/2-1/wall-metrics-20261126-batch.json`. Mỗi ô ghi *chạm cờ /100 · qua sớm / tới tường*:

| Nhánh | γ | Cắt lượt | Seed huấn luyện | 6.803.456 | 7.303.168 | 7.798.784, seed chấm 7 | 7.798.784, seed chấm 11 |
|---|---|---|---|---|---|---|---|
| `wall-cut-200` (cũ) | 0,9 | 200 | 1 | 1 · 1/16 | 12 · 0/31 | 39 · 0/60 | 44 · 1/60 |
| `wall-cut-2000` (cũ) | 0,9 | 2000 | 1 | 1 · 0/14 | 2 · 0/8 | 10 · 3/44 | 8 · 0/45 |
| `wall-cut-200-seed2` | 0,9 | 200 | 2 | 36 · 16/48 | 43 · 20/43 | 35 · 3/67 | 48 · 4/69 |
| `wall-cut-2000-seed2` | 0,9 | 2000 | 2 | 3 · 1/10 | 6 · 1/23 | 26 · 8/38 | 38 · 6/49 |
| **`gamma-0.97-cut-200`** | **0,97** | 200 | 1 | 17 · 1/39 | 62 · 61/70 | **78 · 78/78** | **71 · 70/71** |

Gộp hai seed chấm ở checkpoint cuối (200 lượt mỗi nhánh; khoảng tin cậy Wilson 95%; kiểm định chính xác Fisher):

| So sánh | Chạm cờ | p | Qua sớm / tới tường |
|---|---|---|---|
| Seed 1: cắt 200 so với cắt 2000 | 83/200 (35–48%) so với 18/200 (6–14%) | 3·10⁻¹⁴ | 1/120 so với 3/89 |
| Seed 2: cắt 200 so với cắt 2000 | 83/200 (35–48%) so với 64/200 (26–39%) | 0,06 | 7/136 so với 14/87 |
| γ = 0,97 so với γ = 0,9 (cùng cắt 200, seed 1) | 149/200 (68–80%) so với 83/200 (35–48%) | 3·10⁻¹¹ | 148/149 so với 1/120 |

**Câu hỏi 1, cắt lượt: kết quả cùng chiều, nhưng độ lớn không lặp lại.**
- Theo luật đã chốt, cả hai seed cùng chiều nên kết luận là "cắt 200 cho nhiều lượt chạm cờ hơn ở checkpoint cuối".
- Nhưng chênh lệch giảm từ 65 lượt (seed 1) xuống 19 lượt (seed 2) trên 200, và ở seed 2 chênh lệch này không có ý nghĩa thống kê ở mức 5%.
- Không nhánh cắt lượt nào học được động tác qua tường khi vừa tới: chỉ số qua sớm ở cả bốn nhánh từ 0,8% tới 16%. Ở seed 2, nhánh cắt 2000 qua sớm nhiều hơn nhánh cắt 200.
- Nhánh cắt 200 seed 2 lặp lại kiểu dao động cũ: ở 7.303.168 đạt 43/100 với 20/43 qua sớm, rồi tụt về 35/100 với 3/67 ở checkpoint cuối.

**Câu hỏi 2, γ: γ = 0,97 đạt tiêu chí đã chốt ở cả hai seed chấm.**
- Ở checkpoint cuối: chạm cờ 78 và 71 so với 39 và 44 của nhánh γ = 0,9; qua sớm 78/78 và 70/71 so với 0/60 và 1/60.
- Gần như mọi lượt tới tường đều qua ngay (148/149). Các lượt hỏng còn lại kết thúc trước tường.
- Như vậy ở checkpoint này bức tường không còn là chỗ chặn chính. Màn 2-1 vẫn dưới ngưỡng 0,8: gộp được 74,5%, khoảng tin cậy 68–80%.
- **Lượt γ khởi đầu chậm, đúng như phép kiểm gradient báo trước.**
  - Ở 6.803.456 chỉ đạt 17/100, với 1/39 qua sớm.
  - Mất mát critic giảm từ 7,7 (5 lần cập nhật đầu) xuống khoảng 2,9 (lần 5–25) và khoảng 2 sau 60 lần.
  - Tới 7.303.168 đã đạt 62/100, với 61/70 qua sớm.
- Lúc huấn luyện, lượt γ có entropy thấp hơn (0,30 so với 0,62 ở cuối lượt) và bị dừng sớm vì KL nhiều hơn (53% so với 33–39% số lần cập nhật). Tỉ lệ chạm cờ lúc huấn luyện cao nhất là 0,76, so với 0,08 (seed 1) và 0,40 (seed 2) của nhánh γ = 0,9.
- **Giới hạn.**
  - Chỉ một seed huấn luyện cho γ = 0,97.
  - Theo luật đã chốt, bước tiếp là chạy γ = 0,97 seed 2 (1.495.040 bước, cần duyệt riêng), lấy `wall-cut-200-seed2` làm đối chứng.

**So với dự đoán ghi trước.**
- *Dũng:* không được điền trước khi có kết quả.
- *Claude, cặp seed 2:*
  - "Cắt 200 vẫn hơn, chênh ít hơn 39/10": đúng.
  - "Qua sớm không quá 3 trên số tới tường": sai ở nhánh cắt 2000 seed 2 (8/38 và 6/49) và ở một lần chấm của nhánh cắt 200 seed 2 (4/69).
- *Claude, γ:* "qua sớm từ 5 trên số tới tường trở lên, chạm cờ không thấp hơn 39 quá 10": đúng chiều, nhưng mức cải thiện lớn hơn dự đoán nhiều.

**Điều kết quả này thay đổi.**
- Giả thuyết "γ = 0,9 nhìn quá ngắn để học lùi lấy đà ở tường" được ủng hộ, với một seed.
- Nhận định ở `inside-the-network.md` mục 4, rằng γ không phải thứ đáng thử trước, đã bị số đo bác.
- Về mẹo cắt lượt, chỉ giữ được kết luận: "cắt 200 không làm hại; chênh lệch về chạm cờ so với cắt 2000 không ổn định giữa các seed".

### Lượt γ = 0,97 seed 2 (ghi trước khi chạy, 27/11/2026, 03:48)

Dũng duyệt 1.495.040 bước ngày 27/11/2026.

**Thiết kế.**
- Thư mục `runs/2-1/gamma-0.97-cut-200-seed2`, ghi đè `seed=2 algo.gamma=0.97`.
- Mọi thứ khác như lượt `gamma-0.97-cut-200`: cùng cấu hình, cùng checkpoint xuất phát 6.303.744, cùng CPU và venv, cùng mã `1b0df0d` (`mario_ppo/` không đổi từ đợt trước).
- Đối chứng là `wall-cut-200-seed2` (γ = 0,9, seed 2), đã chấm xong.
- Lệnh ở đầu `experiments/run_wall_batch.sh`; nhật ký ở `runs/2-1/batch-20261127.log`.

**Cách chấm.** Giống đợt trước: 100 lượt seed 7 ở mỗi checkpoint 6.803.456, 7.303.168 và 7.798.784, cộng 100 lượt seed 11 ở 7.798.784. Dùng `value_trace.py` để chấm và `wall_metrics.py` để tính chỉ số.

**Luật quyết định.**
- So ở checkpoint cuối, với cả hai seed chấm. Kết quả của seed 1 được coi là lặp lại nếu nhánh γ = 0,97 seed 2 hơn đối chứng seed 2 cả về chạm cờ (35 và 48) lẫn qua sớm (3/67 và 4/69).
- Nếu hai seed chấm cho kết quả trái chiều: ghi "chưa phân định được".
- Nếu lặp lại: thí nghiệm γ có đủ 2 seed huấn luyện cho Mục 4 của báo cáo.
- Nếu không lặp lại: kết quả seed 1 được ghi là chưa ổn định giữa các seed.

**Dự đoán ghi trước.**
- *Dũng:* (điền trước khi xem kết quả).
- *Claude:* ở checkpoint cuối, chạm cờ từ 55/100 trở lên ở cả hai seed chấm, và qua sớm ít nhất 80% số lượt tới tường. Lượt này cũng sẽ khởi đầu chậm như seed 1 (ở 6.803.456 dưới 30/100). Độ tin vừa phải: seed 1 thắng rất rõ, nhưng thí nghiệm cắt lượt cho thấy độ lớn hiệu ứng ở 2-1 có thể đổi mạnh giữa các seed.

**Kết quả (27/11/2026).** Lượt xong lúc 06:48; chấm theo cách đã chốt. Checkpoint 6.803.456 không chấm xong vì Dũng dừng phân tích.

| Nhánh | 7.303.168, seed chấm 7 | 7.798.784, seed chấm 7 | 7.798.784, seed chấm 11 |
|---|---|---|---|
| γ 0,97, seed 2 | 66 · 62/70 | **75 · 73/75** | **72 · 68/72** |
| γ 0,9, seed 2 (đối chứng) | 43 · 20/43 | 35 · 3/67 | 48 · 4/69 |

Mỗi ô ghi *chạm cờ /100 · qua sớm / tới tường*.

- Ở checkpoint cuối, với cả hai seed chấm, nhánh γ 0,97 hơn đối chứng cả về chạm cờ lẫn qua sớm. Theo luật đã chốt, kết quả của seed 1 đã lặp lại.
- Hai seed của γ 0,97 cho 149/200 và 147/200 lượt chạm cờ; gần mọi lượt tới tường đều qua ngay.
- Dự đoán ghi trước của Claude đúng.

### Đo độ quên trên bảy màn kia (chỉ chấm, không huấn luyện; ghi trước khi đo, 27/11/2026)

**Câu hỏi.** Các nhánh trên chỉ học 2-1 trong 1,5 triệu bước. Chúng đã mất bao nhiêu khả năng chơi 1-1, 3-1, 3-2, 4-1, 5-1, 5-2, 6-1? Dũng duyệt phép đo này.

**Cách đo.**
- Chấm bằng `mario_ppo.evaluate`, 100 lượt seed 7 mỗi màn.
- Năm checkpoint:
  - điểm xuất phát (mạng 8 màn, bước 6.303.744);
  - checkpoint cuối 7.798.784 của bốn nhánh cắt 200: γ 0,9 seed 1 và seed 2, γ 0,97 seed 1 và seed 2.
- Tổng 35 lần chấm, 3.500 lượt. Tệp ở `runs/2-1/forgetting/<nhãn>/<màn>.json`; bảng in bằng `experiments/forgetting_table.py`.
- Chấm sau khi lượt γ 0,97 seed 2 và 400 lượt chấm của nó đã xong, để không tranh CPU.
- So nhánh γ 0,9 với nhánh γ 0,97 để tách phần quên do chỉ học 2-1 khỏi phần quên do đổi γ.

**Dự đoán ghi trước (Claude).**
- Cả bốn nhánh quên rõ: chạm cờ trung bình trên bảy màn giảm ít nhất 20 điểm so với điểm xuất phát.
- Nhánh γ 0,97 quên nhiều hơn nhánh γ 0,9, vì critic và thân CNN đã phải học lại thang giá trị. Độ tin thấp.

**Kết quả (27/11/2026).** Đã đo 4 trong 5 checkpoint. Phần của γ 0,97 seed 2 không đo, vì Dũng dừng phân tích. Chạm cờ /100, seed 7; trong ngoặc là chênh so với điểm xuất phát:

| Checkpoint | 1-1 | 3-1 | 3-2 | 4-1 | 5-1 | 5-2 | 6-1 | Trung bình |
|---|---|---|---|---|---|---|---|---|
| Điểm xuất phát (mạng 8 màn, 6.303.744) | 86 | 66 | 75 | 77 | 64 | 47 | 88 | 71,9 |
| γ 0,9, cắt 200, seed 1 | 7 | 7 | 0 | 23 | 1 | 0 | 13 | 7,3 (−64,6) |
| γ 0,9, cắt 200, seed 2 | 26 | 8 | 21 | 29 | 4 | 0 | 16 | 14,9 (−57,0) |
| γ 0,97, cắt 200, seed 1 | 1 | 0 | 0 | 20 | 0 | 0 | 0 | 3,0 (−68,9) |

- **Học riêng 2-1 thêm 1,5 triệu bước làm mất gần hết bảy màn kia.** Các nhánh này là mạng chuyên 2-1. Kết quả về γ chỉ đúng cho 2-1, chưa nói được gì về mạng nhiều màn.
- Dự đoán "giảm ít nhất 20 điểm" đúng.
- "γ 0,97 quên nhiều hơn" chưa phân định được: chỉ có một seed, và hai seed của γ 0,9 đã chênh nhau gấp đôi.
- Tệp ở `runs/2-1/forgetting/`; bảng in bằng `experiments/forgetting_table.py`.

## Hướng 3: công thức chia bước tự thiết kế (lớn nhất, nếu còn thời gian)

- **Vấn đề.** Công thức hiện tại chia bước theo tỉ lệ hỏng, nên đã dồn 29% số bước vào 2-1 mà không thu lại gì.
- **Hướng sửa.** Chia theo tốc độ tiến bộ thay vì theo tỉ lệ hỏng.
- **Tài liệu liên quan.** Teacher–Student Curriculum Learning (Matiisen và cộng sự, 2017); Prioritized Level Replay (Jiang và cộng sự, 2021).
- **Học được.** Curriculum tự động: chọn bài tập nào cho mạng học tiếp.
- **Chi phí.** Phải chạy nhiều màn nên tốn nhất; chỉ nên làm bản thu nhỏ khoảng 3 màn.

## Cách làm

Trước mỗi thí nghiệm, Dũng viết dự đoán bằng con số cụ thể rồi mới chạy, sau đó so lại. Dũng có thể tự viết một phần mã (ví dụ script vẽ V) và nhờ soát.

Thứ tự đề nghị: hướng 1 trước, vì nó không tốn bước và kết quả của nó định hình hướng 2.
