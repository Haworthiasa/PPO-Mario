# Một mạng PPO vượt 90% trên nhiều màn Mario trong 10 triệu bước

## V2 — Thử hai màn mới song song từ checkpoint 2.904.064

**Phương án đề xuất ngày 21/11/2026.** Dùng một mạng chung cho **1-1, 4-1, 3-2 và 5-1**, mỗi màn hai trong tám môi trường. Mục tiêu đầu là cùng một checkpoint vượt 90% trên cả bốn màn. Màn thứ năm chỉ mở khi bốn màn đầu đã đạt ổn định và còn đủ bước để luyện. V2 là kế hoạch chưa chạy.

### Vì sao chọn bốn màn này?

- **1-1 và 4-1** đã đạt 99/100 mỗi màn bằng cùng một checkpoint. Cho cả hai xuất hiện ở mọi rollout giúp phát hiện quên sớm. Ở thí nghiệm cũ, sau khoảng 200k bước chỉ luyện 4-1, mạng 1-1 đã tụt từ 19/20 xuống 2/20. [Nhật ký huấn luyện](docs/nhat-ky-trong-tu.md)
- **3-2** là ứng viên có bằng chứng rõ nhất: mạng hiện tại đạt trung vị x=1.002, xa nhất 1.312 và 17/20 lượt chết do địch. **5-1** đặt câu hỏi khác: mạng chết do địch cả 20/20 lượt, trung vị x=275, trung bình chỉ 42 bước/lượt. Thử hai màn song song sẽ cho biết màn nào tiến bộ nhanh và có đáng đầu tư tiếp. Chưa có số liệu huấn luyện riêng để xếp chúng theo độ khó. [Khảo sát 32 màn](runs/1-1+4-1/from-scratch/eval/all-levels/summary.csv)
- Cách chia **2 + 2 + 2 + 2** chạy được ngay với bộ thu dữ liệu hiện tại và giữ nguyên cỡ rollout 4.096. Mỗi màn mới lấy được 1.024 bước mỗi lần cập nhật. Cách này dành một nửa số bước cho hai màn cũ; chỉ đầu tư viết thêm mã chia môi trường nếu số đo cho thấy phần ngân sách đó có thể chuyển sang màn mới. [Bộ thu dữ liệu](mario_ppo/train_joint.py)

### Ngân sách và lịch học

Tính ngân sách theo số bước trên đường huấn luyện tạo ra checkpoint cuối, bao gồm **2.904.064 bước** của [checkpoint ban đầu](runs/1-1+4-1/from-scratch/checkpoints/step_002904064.pt). Chọn `algo.total_steps=9_998_336`, mốc rollout cuối trước trần 10M. Như vậy có thể chạy thêm **7.094.272 bước**. Ghi riêng các bước đánh giá và chi phí của các thí nghiệm cũ. Nếu có rẽ nhánh huấn luyện trong chiến dịch V2, tính cả ngân sách đã tiêu ở nhánh bỏ.

Nạp checkpoint bằng `resume`, dùng thư mục kết quả mới. Vòng lặp sẽ khôi phục mạng, Adam và bộ đếm bước. Với lịch 10M, nếu để `algo.lr=2e-4`, lần cập nhật đầu sẽ dùng khoảng `1,42e-4`, trong khi mốc 2,90M của lịch cũ chỉ quanh `6,4e-6`. **Giá trị khởi thử cho V2** là `algo.lr=7e-5`, `anneal_lr=true`, `ent_coef=0.006` và `ent_coef_final=0.002`. Theo công thức hiện tại, tại thời điểm nạp lại, LR thực tế khoảng `4,97e-5` và hệ số entropy khoảng `0,00484`. Đây là lựa chọn cần kiểm chứng để mạng còn khám phá ở màn mới nhưng hạn chế làm xáo trộn hai màn cũ. Giữ cấu hình quan sát, reward và các thông số PPO còn lại của lượt chung. [Cấu hình gốc](configs/group-1-1-and-4-1.yaml) · [Công thức lịch](mario_ppo/ppo.py)

| Mốc tổng | Cách dùng ngân sách và quyết định |
|---|---|
| **2,90M → khoảng 5,30M** | Luyện bốn màn chia đều. Mỗi màn mới nhận khoảng **600k bước**. Lưu checkpoint mỗi 100k và theo dõi số cờ, vị trí kết thúc, tỉ lệ qua chướng ngại đầu trên từng màn. Dùng mốc 250k bước/màn để xem tiến độ. Ở lượt cũ, 4-1 chỉ có 9/100 lúc nhận 251.904 bước riêng nhưng về sau đạt 99/100. [Số đo ở mốc 251.904](runs/1-1+4-1/from-scratch/eval/4-1/100-seed7_000503808.json) |
| **Khoảng 5,30M → 7,50M** | Nếu cả hai màn mới còn tiến bộ, tiếp tục chia đều. Nếu hai màn cũ liên tục trên 90% nhưng một màn mới chững lại, kiểm tra các vị trí hỏng. Khi cần phân bổ lại, thêm hỗ trợ số môi trường theo màn và thử **1 + 1 + 3 + 3**. Nếu một ứng viên vẫn ít cờ và không cải thiện qua chướng ngại ở nhiều checkpoint, cho thêm khoảng 200–300k bước riêng rồi cân nhắc thay bằng **2-1** sau khi chơi thử. Không so x tuyệt đối giữa các màn. |
| **Khoảng 7,50M → 9.998.336** | Nếu bốn màn đều đạt ít nhất 91/100 trên seed phát triển ở hai checkpoint liên tiếp, thử **2-1** hoặc ứng viên đã có tiến bộ khi chơi chéo. Để mở màn thứ năm với tám môi trường, cần hỗ trợ chia theo màn; có thể dành 3 môi trường cho màn mới, 2 cho màn cũ yếu nhất và 1 cho từng màn còn lại. Nếu bốn màn đầu còn sát ngưỡng, ưu tiên củng cố chúng. |

### Tiêu chí chọn checkpoint và bài tập theo đoạn

1. **Theo dõi và xác nhận.** Năm lượt/màn tại checkpoint dùng để phát hiện bất thường. Ở các mốc quyết định, đánh giá các checkpoint ứng viên bằng 100 lượt/màn, seed 7. Xếp theo **số màn đạt ít nhất 91/100**, rồi xét kết quả thấp nhất trong số các màn đạt. Chốt một checkpoint trước khi xác nhận bằng **200 lượt/màn trên hai bộ seed dành riêng** (ví dụ 101 và 1001, mỗi bộ 100 lượt). Ngưỡng quan sát là **181/200 trên từng màn**; 190/200 là bằng chứng mạnh hơn. Báo cáo riêng kết quả chính thức 20 lượt seed 2026. Tổng quát hóa được đo thêm trên 7-1 và 6-3 đã giữ ngoài huấn luyện.
2. **Khi một lỗi lặp lại.** Trước tiên đo số lượt tới chướng ngại và bao nhiêu lượt vượt được. Nếu cần dùng snapshot, lấy từ nhiều quỹ đạo, kiểm tra việc khôi phục bốn khung ảnh lịch sử và so với đối chứng khởi đầu từ đầu màn ở cùng ngân sách. Chỉ giữ phương pháp nếu tỉ lệ chạm cờ toàn màn cải thiện. Thí nghiệm 2-3 hiện cho thấy thay đổi trạng thái xuất phát có thể đưa kết quả từ 4/40 tới 39/40; tác dụng của việc *huấn luyện* bằng snapshot còn cần kiểm chứng. [Log snapshot](runs/2-3/from-scratch-3m/eval/failure-analysis-002801664/start-from-snapshot.txt)

## V1 — Kế hoạch ban đầu

### Mục tiêu và cách tính ngân sách

Mục tiêu đầu tiên là **một checkpoint vượt 90% trên từng màn trong ít nhất 4 màn**. Màn thứ 5 là mục tiêu mở rộng nếu bốn màn đầu đã ổn định ở khoảng 7,5 triệu bước. Đây là kế hoạch thử nghiệm của nhóm, lập ngày 21/11/2026.

Tính 10 triệu **bước tác tử cộng dồn**, gồm cả 2.904.064 bước đã dùng để tạo [checkpoint chung 1-1 và 4-1](runs/1-1+4-1/from-scratch/checkpoints/step_002904064.pt). Còn tối đa **7.095.936 bước** để mở rộng mạng. Với tốc độ của lượt chung trước, dự trù khoảng 15 giờ huấn luyện tiếp, cộng thời gian đánh giá. Ghi riêng bước huấn luyện và lượt đánh giá. Nếu giữ rollout 4.096 bước và áp dụng trần 10 triệu tuyệt đối, mốc dừng ở cuối rollout là **9.998.336**.

### Căn cứ chọn màn

Checkpoint xuất phát đạt 99/100 trên từng màn 1-1 và 4-1 ở seed 7, và 20/20 trên từng màn ở seed 2026. Khi khảo sát 30 màn còn lại, checkpoint này đạt 0/20 trên mỗi màn. Vì vậy, cần đo cả khả năng giữ kỹ năng cũ và tốc độ học kỹ năng mới. [Nhật ký thí nghiệm](docs/nhat-ky-trong-tu.md) · [Khảo sát 32 màn](runs/1-1+4-1/from-scratch/eval/all-levels/summary.csv)

| Màn | Quan sát trước khi huấn luyện thêm | Vai trò dự kiến |
|---|---|---|
| 3-2 | Trung vị x cuối lượt là 1.002, xa nhất 1.312. Có 17/20 lượt chết do địch. | Chọn làm màn thứ ba để đo tốc độ học cách vượt các cụm địch. |
| 5-1 | Trung vị x là 275. Cả 20 lượt chết do địch, độ dài trung bình 42 bước. | Ứng viên màn thứ tư nếu học 3-2 giúp mạng xử lý địch tốt hơn ở đây. |
| 2-1 | Mạng chung chết do địch cả 20 lượt. | Chơi thử lại cùng 5-1 sau giai đoạn học 3-2. |
| 2-3 | Mạng chung thường kẹt gần x=146. Mạng chuyên của nhóm đạt 173/200 trên hai seed. | Ứng viên cho bài tập nhảy cầu và phục hồi sau khi đạp cá. |
| 1-3 và 4-3 | Trong phép khảo sát đầu lượt, mạng chọn nút A khoảng 95% số quyết định và thường nhảy tại chỗ. | Ví dụ để phân tích hành vi lặp khi gặp cảnh mới. |

Chọn màn dựa trên khả năng **cùng mạng** chạm cờ và mức tiến bộ sau mỗi giai đoạn. Vị trí x giúp tìm chướng ngại cụ thể; độ dài và địa hình khác nhau khiến x tuyệt đối khó dùng để xếp độ khó giữa các màn. [Hành động đầu lượt](runs/1-1+4-1/from-scratch/eval/all-levels/first-400-steps.txt)

### Lịch huấn luyện

| Bước tổng | Phân bổ và thí nghiệm | Quyết định ở cuối giai đoạn |
|---|---|---|
| 2,904M → 5M | Giữ 8 môi trường. Gán 1 môi trường cho 1-1, 1 cho 4-1, 6 cho 3-2. Thu dữ liệu cả ba màn trong mỗi lần cập nhật. | Nếu kết quả đánh giá đủ lượt ở một màn cũ liên tục dưới 90%, tăng phần luyện màn đó. Nếu 3-2 lặp lại một điểm chết, dành tối đa 2 trong 6 môi trường của 3-2 cho đoạn khó. |
| 5M → 7,5M | Chơi thử lại 5-1, 2-1 và 2-3 bằng checkpoint hiện tại. Đối chiếu số cờ, vị trí chết và mức tiến bộ so với khảo sát ban đầu. | Thêm một màn thứ tư khi ba màn đầu đủ ổn định. Ưu tiên 5-1 nếu số lượt chết do địch đã giảm rõ sau khi học 3-2. |
| 7,5M → tối đa 10M | Tiếp tục lấy mẫu từ các màn cũ. Nếu bốn màn đã ổn định trên 90%, thêm màn thứ năm. Nếu một màn còn quanh 75–89%, tập trung ngân sách cho màn đó. | Chọn checkpoint có nhiều màn vượt 90% nhất. Khi bằng nhau, ưu tiên checkpoint có tỉ lệ cao hơn trên màn yếu nhất trong số các màn đạt. |

### Cách luyện một kỹ năng theo đoạn

Thử trước ở chướng ngại mà mạng đã gặp lặp lại. Ví dụ, ở đoạn cầu 2-3, trong số lượt tới khu vực khảo sát, 11/25 lượt có dấu hiệu đạp cá chết gần đó, so với 2/167 lượt không ghi nhận cú đạp. [Dữ liệu từng nhóm tình huống](runs/2-3/from-scratch-3m/eval/failure-analysis-002801664/enemy-hits.txt)

1. Lưu nhiều trạng thái **trước** chướng ngại từ cả lượt thành công và thất bại. Chọn nhiều tốc độ chạy, vị trí cá, độ cao và khoảng cách tới mép cầu. Khi khôi phục cần đồng bộ trạng thái giả lập, lịch sử khung hình và trạng thái của các wrapper liên quan.
2. Dành ban đầu tối đa 2/8 môi trường để luyện đoạn. Sáu môi trường còn lại chơi từ đầu màn. Trong các môi trường luyện đoạn, chính sách hiện tại tự chọn hành động và tạo rollout mới cho PPO.
3. Cho Mario tiếp tục chơi sau cú nhảy để học được cả việc tiếp đất và xử lý chướng ngại kế tiếp. Theo dõi tỉ lệ vượt đoạn và tỉ lệ chạm cờ khi chơi lại từ đầu. Nếu chỉ số ở đoạn tăng mà số cờ cả màn đứng yên, mở rộng phân bố trạng thái xuất phát hoặc giảm phần luyện đoạn.
4. Chỉ dùng điểm xuất phát tùy chỉnh khi train. Mạng nhận các khung hình đã tiền xử lý. Chấm bằng script gốc từ đầu màn theo [quy định của đề](de-bai.md).

### Cách đo thành công và tổng quát hóa

- **Các màn đã huấn luyện.** Đánh giá cùng một checkpoint trên từng màn bằng chính sách lấy mẫu. Dùng một bộ seed để chọn checkpoint, rồi xác nhận bằng seed khác. Trong 200 lượt xác nhận cho mỗi màn, 181/200 là vừa vượt 90% quan sát được. Mục tiêu xác nhận mạnh hơn là 190/200, khi cận dưới Wilson 95% vượt 90% theo mô hình nhị thức. Ghi riêng kết quả 20 lượt seed 2026 theo đề.
- **Các màn giữ ngoài huấn luyện.** Chốt trước hai màn, đề xuất 7-1 và 6-3. Đánh giá từ đầu màn, dùng cùng giao thức và không cập nhật trọng số. So số cờ và quãng đường với checkpoint xuất phát. Nhóm đã khảo sát cả 32 màn nên cần mô tả rõ hai màn này đã từng được đo. Tránh dùng điểm của chúng để chọn cấu hình hoặc checkpoint.
- **Nhật ký.** Với mỗi checkpoint, lưu bước tổng, số bước dành cho mỗi màn, tỉ lệ chạm cờ riêng, số lượt và seed đánh giá. Với chướng ngại cần luyện đoạn, ghi cả số lượt tới được đoạn đó và số lượt chết trong đoạn.

### Hai việc kỹ thuật trước lần chạy đầu

1. Mở rộng [train_joint.py](mario_ppo/train_joint.py) để nhận số môi trường cho từng màn. Hàm hiện tại chia đều và đòi số môi trường chia hết cho số màn, nên chưa chạy được lịch 1 + 1 + 6. Tiếp tục tính GAE theo từng môi trường và ghi thống kê riêng theo màn.
2. Đặt rõ learning rate đầu giai đoạn mở rộng. Cơ chế resume khôi phục trọng số, Adam và bộ đếm bước. Nếu chỉ đổi total_steps từ 3M thành 10M, cùng bước 2.904.064 sẽ làm learning rate theo lịch tăng từ khoảng 6,4 × 10⁻⁶ lên 1,42 × 10⁻⁴. Kiểm tra cả lịch entropy và đánh giá lại 1-1, 4-1 ngay ở các checkpoint đầu để phát hiện quên kỹ năng.

**Thứ tự ưu tiên:** bắt đầu với phân bổ 1 + 1 + 6 và cấu hình CNN, reward hiện có. Chỉ thêm bài tập theo đoạn khi dữ liệu chỉ ra chướng ngại lặp lại. Mốc quyết định đầu tiên là số bước cần để 3-2 vượt 90% trong khi 1-1 và 4-1 vẫn giữ được mức đó.

## Tài liệu đã đối chiếu

- Florensa và cộng sự nghiên cứu cách chọn trạng thái xuất phát để tác tử học tình huống khó tiếp cận. [Reverse Curriculum Generation](https://proceedings.mlr.press/v78/florensa17a.html)
- Jiang và cộng sự nghiên cứu phân bổ lần luyện lại cho những màn còn tiềm năng học. [Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html)
- Hai benchmark cung cấp cách thiết kế phép đo chuyển giao và tổng quát hóa trên màn giữ ngoài huấn luyện. [Gotta Learn Fast](https://arxiv.org/abs/1804.03720) · [Procgen Benchmark](https://proceedings.mlr.press/v119/cobbe20a.html)
