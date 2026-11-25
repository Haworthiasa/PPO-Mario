# Học kỹ năng Mario để vượt nhiều màn trong 10 triệu bước

Ngày 21/11/2026. Phương án chính là **một mạng khởi tạo ngẫu nhiên, học từ đầu**, theo lựa chọn của người dùng. Mục tiêu là số màn cùng một checkpoint vượt được; dùng mức từ 90% theo hướng trong hai bản bigidea. Khả năng chơi màn chưa học được đo riêng.

**Ưu tiên của tôi là thay cách tạo bài học và phân bổ bước trước, rồi mới tăng độ phức tạp của mạng.** Dùng PPO với đầu vào bốn ảnh hiện tại, trộn nhiều màn và các đoạn luyện kỹ năng có trạng thái xuất phát đa dạng, sau đó giảm learning rate để ổn định. CNN residual là ứng viên đáng thử nếu chấp nhận thời gian CPU cao hơn. Chia ảnh thành các vùng hoặc đặt tên các đầu ra là “skill” tự nó chưa tạo ra khả năng chuyển giao.

## 1 Những bằng chứng đang có

| Bằng chứng | Ý nghĩa đối với hướng mới |
|---|---|
| Mạng chung 1-1 và 4-1 đạt 99/100 ở mỗi màn, checkpoint 2.904.064 | Mạng hiện tại đủ sức chứa ít nhất hai màn và có thể học chung. |
| Cùng mạng đạt 0/600 lượt chạm cờ trên 30 màn khác | Chưa có bằng chứng chuyển giao tới mức hoàn thành màn trong giao thức đã đo. Không chứng minh mọi kỹ năng đều không chuyển giao. |
| Chỉ học tiếp 4-1 làm kết quả 1-1 từ 19/20 xuống 2/20 sau khoảng 200k bước | Cần giữ dữ liệu từ các màn/kỹ năng cũ trong quá trình mở rộng. |
| Trong lượt thử nối tiếp tám màn, 3-2 tiến nhanh; 6-1 và 8-1 cũng tiến bộ về x | Đây là tín hiệu chọn ứng viên, nhưng đến từ mạng đã học hai màn, chưa phải kết quả của mạng mới. |
| Mạng chuyên 2-3 đạt 173/200 ở checkpoint được chọn | Một màn có nhiều tình huống nhảy có thể cần ngân sách lớn mà vẫn chưa đạt 90%. |

Nguồn nội bộ: [nhật ký](nhat-ky-trong-tu.md), [khảo sát 32 màn](../runs/1-1+4-1/from-scratch/eval/all-levels/summary.csv), [log thử nối tiếp](../runs/multi-level/restart-test/progress.csv).

Cần phân biệt ba mục tiêu: **giữ được nhiều màn đã luyện**, **áp dụng kỹ năng ở đoạn/màn chưa luyện**, và **học màn mới nhanh hơn**. Một mạng có thể cải thiện mục tiêu thứ ba trước khi chạm cờ được ở mục tiêu thứ hai. Đánh giá chỉ bằng số cờ sẽ bỏ sót tiến bộ này.

## 2 Các phép đo mới

Đã chạy [script chẩn đoán](../experiments/skill_probe.py), lưu [kết quả đầy đủ](../runs/skill-analysis-20261121/probe.json). Thời gian chạy khoảng 46 giây trên CPU, PyTorch 2.15.0, sáu luồng. Không có lượt PPO huấn luyện mới và không ghi checkpoint. Đo tốc độ dùng dữ liệu ảnh tổng hợp; đo phản ứng dùng checkpoint cũ để chẩn đoán, không dùng trọng số đó khởi tạo mạng mới.

### Cơ chế nhảy

Đo từ đầu 1-1, Mario nhỏ, mặt đất bằng, `frame_skip=4`. Các ca chạy phải+B có năm quyết định chạy đà trước lúc nhảy. Chiều cao và quãng đường đo bằng tọa độ để phân tích; tọa độ không được đưa vào mạng.

| Số quyết định giữ A | Đứng tại chỗ cao thêm | Đi phải cao thêm / đi xa | Chạy phải+B cao thêm / đi xa |
|---|---:|---:|---:|
| 1 | 33 px | 35 / 40 px | 42 / 69 px |
| 2 | 43 px | 46 / 54 px | 55 / 93 px |
| 4 | 58 px | 60 / 68 px | 75 / 117 px |
| 8 | 64 px | 68 / 82 px | 86 / 153 px |
| 12 | 64 px | 68 / 82 px | 86 / 153 px |

Đây là các ca cơ học cố định, không phải tỉ lệ thành công trên nhiều địa hình. Đỉnh nhảy lấy tại thời điểm quan sát mỗi bốn khung nên có thể bỏ qua đỉnh thật giữa hai quan sát. Quãng đường được tính tới quan sát tiếp đất.

Kết quả cho thấy cần học cả **thời gian giữ và nhả A**, cùng vận tốc trước lúc nhảy. Giữ A lâu hơn không tăng chiều cao mãi. Bài học “nhảy đúng lúc” cần nối với “chọn độ cao”, “chọn điểm tiếp đất” và “nhả A để chuẩn bị cú tiếp theo”. Bước đạp địch tạo một trạng thái bay mới, cần luyện phục hồi riêng.

### Chi phí kiến trúc

Mỗi số thời gian là trung vị ba nhóm đo sau khởi động. Các mạng IMPALA dưới đây chỉ là ứng viên khởi tạo ngẫu nhiên để đo chi phí, chưa được huấn luyện trong game.

| Mạng | Tham số | Suy luận batch 8 | Forward backward Adam batch 256 |
|---|---:|---:|---:|
| Nature-CNN hiện tại, dense 512 | 1.688.232 | 2,74 ms | 106 ms |
| IMPALA kênh 16/32/32, dense 256 | 1.091.288 | 11,30 ms | 732 ms |
| Cùng IMPALA, gộp lưới 4×4 và toàn ảnh, dense 256 | 239.320 | 9,37 ms | 766 ms |

Gộp theo vùng giảm tham số nhưng phần tích chập vẫn tốn gần như cũ. Không thể lấy tỉ lệ thời gian này nhân thẳng vào thời gian huấn luyện cả lượt: còn giả lập, thu rollout, số epoch thực chạy và đánh giá. Chưa có benchmark toàn bộ vòng PPO với hai ứng viên mới.

### Phụ thuộc vào chuyển động và vị trí ảnh

Trên quỹ đạo ngắn của mạng cũ ở 1-1 và 4-1, thay bốn ảnh lịch sử bằng bốn bản của ảnh cuối làm hành động có xác suất cao nhất trùng với ban đầu chỉ khoảng **38% và 36%**. Điều này ủng hộ việc giữ thông tin chuyển động và đồng bộ frame stack khi luyện từ snapshot.

Dịch toàn ảnh hai pixel sang phải làm độ trùng hành động còn khoảng 66–68% ở hai màn đó. Đây là dấu hiệu cần kiểm tra độ bền trước dịch chuyển nhỏ. Phép biến đổi cũng tác động tới Mario, địa hình và HUD, nên chưa tách được nguyên nhân hoặc chứng minh mạng học thuộc màn. Che mười hàng trên cùng cho độ trùng khoảng 80–84%; chưa có căn cứ bỏ HUD để chữa lỗi chính.

Ở 1-3, xác suất trung bình chọn hành động có A khoảng 99,99% trên 384 quan sát đầu lượt. Mạng có một hành vi thiên lệch mạnh trong cảnh lạ; kết quả này không phải đánh giá khả năng của kiến trúc mới.

Tổng số bước môi trường chẩn đoán là **2.331**, gồm 372 bước cho các ca nhảy và 1.959 bước lấy quan sát chính sách. Các bước này được ghi riêng, không phải bước huấn luyện mạng mới.

## 3 Đổi cấu trúc mạng có đáng không

Nature-CNN hiện tại có lớp đầu kernel 8 stride 4, sau ba convolution còn bản đồ 7×7. Trường nhìn lý thuyết của mỗi ô cuối là 36×36 pixel đầu vào, bước lưới tám pixel; lớp dense phía sau mới phối hợp các vùng toàn ảnh. Lớp dense 512 chứa **1.606.144 tham số, khoảng 95,1% toàn mạng**.

Điều này gợi ý hai hạn chế có thể kiểm tra: giảm độ phân giải sớm có thể làm khó việc nhận mép hố và vật nhỏ; lớp dense gắn đặc trưng với vị trí tuyệt đối có thể khuyến khích dựa vào bố cục quen. Cả hai là giả thuyết kiến trúc, chưa phải nguyên nhân đã xác nhận bằng thí nghiệm học.

CNN residual kiểu IMPALA đáng thử vì có bằng chứng trên Procgen rằng kiến trúc và kích thước mạng ảnh hưởng tới hiệu quả dùng mẫu và tổng quát hóa. Kết quả đó không xác định số bước cần thiết hoặc kiến trúc tốt nhất cho SMB của nhóm. [Procgen](https://proceedings.mlr.press/v119/cobbe20a.html)

| Hướng cấu trúc | Đánh giá trong điều kiện hiện tại |
|---|---|
| Giữ Nature-CNN, đổi curriculum và bộ lấy mẫu | Phương án chính ít rủi ro nhất trên CPU; mạng mới vẫn khởi tạo từ đầu. |
| IMPALA nhỏ hoặc residual ít block hơn | Ứng viên đối chứng mạnh nếu thời gian máy cho phép. Bản đã đo tốn gần bảy lần chi phí cập nhật tổng hợp. |
| Mạng lớn hơn chỉ bằng tăng dense | Ưu tiên thấp; phần lớn tham số hiện đã nằm ở dense. |
| Transformer, nhiều chuyên gia hoặc hệ phân cấp lớn | Quá nhiều thành phần phải cùng học trong 10M; chưa có bằng chứng đây là nút thắt cần giải trước. |
| LSTM có hidden state xuyên các lần `act` | Cần xử lý reset hidden state và huấn luyện theo chuỗi. Evaluator hiện không gọi API reset hidden; chưa phù hợp để thay vào trực tiếp. |

Nếu chỉ trần số bước là quan trọng và có thể dành thêm nhiều thời gian CPU, tôi sẽ nâng ưu tiên residual. Nếu cần một lượt chạy có xác suất hoàn tất cao trên máy hiện tại, tôi giữ Nature-CNN và tập trung vào bài học. Phép đo tốc độ chưa quyết định được mạng nào dùng ít bước hơn.

## 4 Chia vùng trên cấu trúc

Nên phân biệt chia **vùng ảnh** với chia **chức năng của mạng**.

Với vùng ảnh, một nhánh toàn cảnh cần thấy đường đi, trần thấp, điểm đáp và chướng ngại tiếp theo. Các vùng cục bộ có thể nhấn mạnh chân Mario, mép phía trước và vùng có thể tiếp đất. Cùng một encoder nên dùng chung đặc trưng, sau đó gộp chúng; bước sau mới cân nhắc attention học cách chọn vùng.

Không nên cắt cứng chỉ giữ nửa phải hoặc chỉ giữ một ô cố định quanh vị trí giả định của Mario. Trước khi camera cuộn Mario thay đổi vị trí trên ảnh; khi bay độ cao thay đổi; né địch có thể cần lùi. Vùng gần giúp phản xạ nhưng không đủ để chọn một cú nhảy có điểm đáp tốt.

Với chức năng, actor có thể dùng chung thân mạng với các đầu phụ học chuyển động từ cặp ảnh, hoặc phân biệt pha lên/xuống/tiếp đất bằng tín hiệu rút từ ảnh. Chỉ thêm khi chỉ ra được lỗi nhận thức và có phép đo hữu ích. Không tạo một mạng riêng cho từng màn rồi dùng mã màn để chọn: cách đó không kiểm chứng ý tưởng kỹ năng dùng lại và không phù hợp đầu vào ảnh duy nhất.

**Thứ tự thử:** toàn ảnh hiện tại → thêm cách gộp đặc trưng không gian → attention hoặc đầu phụ khi có bằng chứng. Kết quả timing cho thấy gộp vùng giúp giảm số tham số, chưa cho thấy tăng chất lượng hay giảm thời gian CPU.

## 5 Kỹ năng nên được định nghĩa như thế nào

Một kỹ năng có thể dùng lại phải có điều kiện hình học và động học, cùng tiêu chí kết thúc. Ví dụ: “vượt khoảng trống rồi đáp an toàn khi tốc độ ban đầu thay đổi”, thay vì “đến x=680 thì nhảy”.

| Nhóm kỹ năng | Cần thay đổi trong bài tập | Tiêu chí hoàn thành |
|---|---|---|
| Nhảy thấp vượt vật cản | Khoảng cách, tốc độ chạy, độ cao vật cản | Vượt và còn kiểm soát được sau tiếp đất |
| Nhảy xa qua hố | Đà chạy, mép hố, độ rộng và độ cao điểm đáp | Đáp được và sống thêm một đoạn |
| Nhảy dưới trần thấp | Khoảng hở trên đầu và cự ly | Không bị chặn rồi rơi hoặc va địch |
| Đạp hoặc né địch | Vị trí và pha di chuyển của địch | Qua được tình huống, không chỉ có điểm hạ địch |
| Phục hồi sau đạp địch | Độ cao, hướng bay, vật cản kế tiếp | Chọn lại được điểm đáp |
| Chờ rồi đi tiếp | Nhịp địch, vật chuyển động, khoảng trống | Chờ đúng lúc và vẫn tiếp tục tiến |
| Nhảy liên tiếp | Khoảng cách giữa hai lần đáp và yêu cầu nhả A | Hoàn thành cả chuỗi |

Các dải giữ A 1/2/4/8 quyết định từ phép đo có thể dùng làm nhóm kiểm tra hành vi. Chúng không phải lệnh bắt buộc hoặc bộ đáp án cho mọi địa hình.

Học kỹ năng không bắt buộc phải có nhiều mạng con. Giai đoạn đầu nên dùng **một policy chung**, cho nó gặp các bài tập có cấu trúc và nhận kết quả vượt đoạn. Hệ options học cả chính sách nội bộ và điều kiện kết thúc là một hướng nghiên cứu có cơ sở, nhưng thêm gánh nặng chọn kỹ năng và học kết thúc. [Option-Critic](https://arxiv.org/abs/1609.05140)

Chưa nên đổi sang macro-action “nhảy dài tám bước” trong lượt chính. Nó có thể làm mất khả năng sửa cú nhảy giữa chừng, thay ngữ nghĩa tập hành động của evaluator và khiến số bước khó so sánh. Học thời gian cam kết hành động có nghiên cứu riêng, nhưng kết quả TempoRL không phải bằng chứng rằng có thể gắn trực tiếp vào PPO này. [TempoRL](https://proceedings.mlr.press/v139/biedenkapp21a.html)

## 6 Curriculum để học cách xử lý tình huống

Từ đầu, cho policy tiếp xúc nhiều màn mặt đất, thay vì hoàn tất một màn rồi chuyển hẳn sang màn khác. Một nhóm ứng viên hợp lý là 1-1, 4-1, 3-2, 6-1, 2-1 và 8-1. Đây là danh sách thăm dò dựa trên dữ liệu cũ, không phải thứ hạng độ khó đã xác nhận cho mạng mới. 5-1 có thể thay một ứng viên sau khi quan sát khả năng học mới.

Sau khi có quỹ đạo, lấy nhiều snapshot trước các tình huống có thể học, từ chính lượt chạy mới. Dành khoảng **2/8 môi trường** cho luyện đoạn và phần còn lại cho chơi từ đầu. Khi chưa có snapshot phù hợp, cả tám môi trường chơi từ đầu. Khoảng 25% là mức bắt đầu để thử, không phải tỉ lệ tối ưu đã biết.

Các snapshot của cùng nhóm kỹ năng phải khác nhau về màn, tốc độ, khoảng cách tới mép, độ cao, pha địch và lịch sử ảnh. Bắt đầu ở những trạng thái policy đã có khả năng xử lý một phần, sau đó kéo điểm xuất phát xa dần và nối với tình huống tiếp theo. Cách chọn điểm xuất phát này có cơ sở từ nghiên cứu reverse curriculum; hiệu quả trong Mario vẫn cần đo. [Reverse Curriculum](https://proceedings.mlr.press/v78/florensa17a.html)

API `nes-py` đã cài chỉ có `_backup()` lưu nội bộ và `_restore()` khôi phục; chúng không trả hoặc nhận một blob snapshot để xếp vào ngân hàng Python. Một ngân hàng nhiều trạng thái tùy chọn chưa có sẵn. Bản triển khai đầu nên để mỗi worker tự thu một điểm xuất phát, luyện từ điểm đó rồi định kỳ thay bằng tình huống khác; các worker có thể luân phiên vai trò thu và luyện. Muốn lưu lâu dài hoặc chuyển snapshot giữa worker cần bổ sung cách tuần tự hóa trạng thái hay tái chạy quỹ đạo nguồn; các bước tái chạy để thu bài phải tính vào ngân sách. Không coi việc sao chép RAM là đủ để sao chép toàn bộ giả lập.

Không chỉ luyện lặp lại đúng một save-state. Dữ liệu cũ ở 2-3 cho thấy chín trạng thái gần cùng vị trí cho kết quả từ 4/40 tới 39/40 mà không cập nhật trọng số. Đây là khác biệt do trạng thái xuất phát, không phải thành quả học kỹ năng.

Các yêu cầu triển khai quan trọng:

- Lưu và khôi phục trạng thái giả lập, bốn khung lịch sử, bộ nhớ FrameSkip và trạng thái wrapper liên quan; xử lý lại thống kê episode, phần thưởng và giới hạn đứng yên. Phép thử restore phải tái tạo được quan sát và vài chuyển tiếp sau đó với cùng hành động.
- Snapshot không đồng nghĩa với kết thúc lượt. Nếu giới hạn bài tập chỉ là cắt thời gian, cần bootstrap critic đúng; chết thực sự mới dùng terminal. Không tự động đặt mọi lần vượt đoạn thành terminal và thưởng lớn.
- PPO lấy rollout mới bằng policy hiện tại từ các trạng thái được chọn. Không lấy các transition cũ trong thư viện snapshot rồi dùng như dữ liệu PPO mới.
- Sau khi vượt chướng ngại, tiếp tục chơi đủ lâu để học điểm đáp và chướng ngại kế tiếp. Kết quả cuối vẫn chấm từ đầu màn.
- Tọa độ, thông tin địch hoặc nhãn màn dùng để phân tích và quản lý bài tập không được nối vào đầu vào policy. Snapshot chỉ có vai trò khởi tạo môi trường khi train. Quyền dùng save-state trong bài nộp cần được xác định riêng nếu áp dụng quy định của giảng viên.

Nếu không dùng được snapshot, vẫn có thể trộn nhiều màn, giữ NOOP ngẫu nhiên và phân bổ nhiều lượt hơn cho màn có tiến bộ. Tuy nhiên, luyện chướng ngại ở cuối màn sẽ tiếp tục tốn bước đi lại từ đầu.

Ưu tiên bài còn học được, có tiến bộ, hoặc có dấu hiệu quên; giữ một phần lấy mẫu đều để không bỏ quên các bài khác. Không dồn toàn bộ vào nơi chết nhiều nhất, vì đó có thể là tình huống hiện chưa tiếp cận được. Đây là nguyên tắc gần với PLR, chưa phải đã cài thuật toán PLR. [Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html)

## 7 Trọng số cố định hay giảm dần

| Đại lượng | Hướng đề xuất |
|---|---|
| Trọng số mạng neural | Cập nhật toàn mạng khi train, cố định khi đánh giá. Chưa đóng băng thân CNN hoặc tách từng skill cố định. |
| Learning rate | Giữ khả năng học trong giai đoạn nhiều tình huống mới, rồi giảm rõ ở giai đoạn cuối. Không giảm về gần 0 khi vẫn dự định thêm màn. |
| Hệ số entropy | Giảm có lịch để cuối lượt lấy mẫu ổn định hơn; không tăng lại mạnh chỉ vì một màn mới khó. Đo riêng trên từng màn/bài tập. |
| Trọng số reward | Giữ ổn định trong lượt đối chứng đầu để dễ diễn giải. Chỉ thêm shaping có mục tiêu và kiểm tra hành vi khai thác thưởng. |
| Tỉ lệ lấy mẫu màn và skill | Thay đổi theo tiến bộ và mức quên; có mức lấy mẫu tối thiểu cho các kỹ năng cần giữ. |
| Hệ số loss phụ | Bắt đầu nhỏ nếu cần; tách khỏi reward nhiệm vụ để đo tác dụng riêng. Không mặc định giảm mọi hệ số cùng learning rate. |

Một lịch khởi thử cho mạng **từ đầu** là warmup ngắn trong khoảng 100k bước tới `2e-4`, duy trì khoảng `1e-4–2e-4` khi mở rộng bài học tới khoảng 6M, rồi giảm về `1e-5–2e-5` trong phần còn lại. Hệ số entropy có thể bắt đầu `0.01`, giảm tới khoảng `0.002`, và chỉ hạ thêm nếu đánh giá lấy mẫu cho thấy vẫn quá nhiễu. Đây là điểm xuất phát cần kiểm chứng, không phải lịch đã được chứng minh tốt hơn lịch tuyến tính cũ.

Quan trọng hơn các con số là tránh hai cực: LR lớn mãi làm chính sách cuối dao động; LR nhỏ quá sớm làm màn vào muộn khó học. Dữ liệu cũ thay đổi cả LR, entropy, checkpoint xuất phát và tập màn nên không thể quy hết chênh lệch thành công cho riêng LR.

Trong mã hiện tại, `scheduled_ent_coef` suy tiến độ từ LR của optimizer và chỉ cho phép `anneal_lr=true`. Vì vậy muốn dùng plateau, warmup hoặc LR thích nghi thì phải **tách bộ đếm bước của lịch entropy khỏi LR**. Không chỉ đổi `anneal_lr=false` rồi giữ nguyên cấu hình entropy.

Log restart có 91/91 lần cập nhật dừng trước mức tối đa 160 minibatch, số thực tế nhỏ nhất/trung vị/lớn nhất là **16/32/64**, KL trung bình 0,0276 và cao nhất 0,0887. Đây là cảnh báo để đo KL theo từng màn; KL gộp có thể che việc một màn đổi quá mạnh. Không nên chỉ tắt chặn KL để ép đủ epoch.

## 8 Gamma và reward có thể quan trọng hơn tăng số lớp

`gamma=0.9`, `gae_lambda=1.0` là cấu hình hiện tại. Trọng số của một reward sau k quyết định là gamma mũ k:

| Gamma | Sau 10 bước | Sau 30 bước | Sau 50 bước |
|---|---:|---:|---:|
| 0,90 | 0,349 | 0,042 | 0,005 |
| 0,97 | 0,737 | 0,401 | 0,218 |
| 0,99 | 0,904 | 0,740 | 0,605 |

Một cú nhảy đơn trong phép đo kéo dài khoảng 6–13 quyết định. Do đó gamma 0,9 không bỏ qua toàn bộ hậu quả của một cú nhảy; hạn chế có thể lộ rõ hơn khi phải chuẩn bị và nối nhiều chướng ngại. Mạng đã giải được hai màn với gamma này, nên không có căn cứ gọi nó là lỗi cài đặt.

Tôi xếp phép thử **gamma 0,90 so với 0,97** khá cao; chỉ thử 0,99 khi critic vẫn ổn. Để phân biệt riêng ảnh hưởng gamma, lần đối chứng đầu giữ lambda và các lựa chọn khác. Tăng gamma làm đổi quy mô return và loss critic; phải xem explained variance, loss và chuẩn gradient theo từng màn. Chuẩn hóa advantage không tự chuẩn hóa value target. Nếu sau đó đổi lambda xuống 0,95–0,98 thì phải coi đó là phép thử khác: độ dài trace GAE phụ thuộc tích gamma nhân lambda.

Reward hiện tại có trọng số `[1, 1, 2, 0, 0]` và thưởng cờ 2.0. Tôi giữ chúng cho lượt đầu, tránh thưởng trực tiếp mỗi lần nhảy hoặc thưởng lớn khi đạp địch. Những thưởng đó có thể tạo nhảy lặp, săn điểm hoặc chọn cú đạp khiến hỏng điểm đáp. Tiến độ bài học nên đo bằng qua chướng ngại và sống tiếp, thay vì chỉ tăng số lần bấm A.

## 9 Chứng minh kỹ năng thay vì chỉ học thuộc

Mỗi nhóm kỹ năng cần các tình huống luyện và tình huống kiểm tra khác nhau. Chia theo **đoạn hoặc quỹ đạo nguồn**, không chia ngẫu nhiên các frame liền kề. Snapshot cùng một quỹ đạo không được xuất hiện ở cả hai phía rồi gọi là chuyển giao.

Đo tối thiểu bốn đại lượng: tỉ lệ vượt đoạn, tỉ lệ sống sau điểm đáp, số cờ từ đầu màn, và số bước để học thêm một tình huống mới. Giữ riêng một màn mặt đất chưa train, chẳng hạn 7-1 hoặc 5-1, để đo chuyển giao cùng nhóm; thêm 6-3 để khảo sát nhóm platform khác. Các màn này từng được khảo sát bởi mạng cũ nên phải mô tả đúng là không huấn luyện trong lượt mới, không gọi là chưa từng nhìn số liệu.

Ví dụ, mô hình học qua hố ở 1-1 và 4-1 rồi vượt được nhóm hố ở màn giữ ngoài huấn luyện với nhiều tốc độ ban đầu là bằng chứng tốt hơn việc tăng điểm ngay tại snapshot đã lặp lại. Một hành vi tương ứng với “skill” phải thay đổi đúng theo cảnh; tên đầu mạng và bản đồ attention không đủ chứng minh.

Độ chính xác cục bộ cần rất cao. Nếu mười chướng ngại độc lập, mỗi chỗ qua 95%, tỉ lệ qua cả chuỗi chỉ khoảng 60%; mỗi chỗ 99% thì khoảng 90%. Thực tế các lỗi không độc lập, nhưng phép tính cho thấy vì sao “qua bài nhảy 90%” có thể vẫn rất yếu khi nối cả màn.

Với checkpoint cuối, dùng hành động lấy mẫu như evaluator gốc. Theo dõi nhanh bằng ít lượt, chọn ứng viên bằng seed phát triển rồi chốt trước khi xác nhận 200 lượt/màn trên seed riêng. Báo cáo cả tỉ lệ và khoảng tin cậy: 180/200 mới đạt 90% quan sát được, chưa chứng minh xác suất thật ít nhất 90%. 190/200 có cận dưới Wilson 95% trên 90% dưới mô hình nhị thức. Vẫn báo cáo riêng 20 lượt seed 2026 theo đề. Không đổi sang argmax để tăng kết quả.

## 10 Phân bổ tối đa 10 triệu bước

Ngân sách chính tính như bộ đếm huấn luyện của repo: tổng chuyển tiếp `T × N` của các môi trường, không phải số khung NES. Frame skip, bước NOOP trong reset và bước đánh giá cần ghi riêng. Với rollout 4.096, mốc cuối không vượt trần là **9.998.336**.

| Mốc gần đúng | Việc chính | Quyết định |
|---|---|---|
| 0–0,8M | Khởi tạo ngẫu nhiên, tiếp xúc nhóm sáu màn ứng viên, thu lỗi và các snapshot khả dụng | Kiểm tra có học tiến lên, nhả/giữ A và critic ổn; chưa loại màn chỉ vì chưa có cờ. |
| 0,8–3M | Trộn chơi toàn màn với tối đa 25% luyện đoạn; đa dạng hóa trạng thái theo nhóm kỹ năng | Chọn khoảng 4–6 ứng viên theo cờ, qua chướng ngại, tiến bộ và độ tin cậy của số đo. |
| 3–7,5M | Tập trung các màn còn tiềm năng, giữ phần dữ liệu chống quên, nối các skill thành chuỗi | Tái xét màn chững lại; không xếp độ khó bằng x tuyệt đối giữa các màn khác nhau. |
| 7,5–9,998M | Giảm LR, tăng chơi từ đầu màn và củng cố các màn sát ngưỡng | Tránh mở nhóm cơ chế mới quá muộn; chọn một checkpoint cho toàn bộ màn. |

Kế hoạch này cần sửa bộ thu hiện tại: `assign_levels` chỉ hỗ trợ chia đều và số môi trường phải chia hết số màn. Chưa chạy được sáu màn cộng hai slot luyện đoạn chỉ bằng sửa YAML. Cần phân bổ không đều hoặc bộ chọn bài khi reset, nhãn bài theo transition, thống kê và KL theo màn, lưu curriculum và RNG để nối tiếp có thể tái tạo.

Với khoảng 25% bước cho luyện đoạn, 10M còn xấp xỉ 7,5M bước chơi toàn màn. Chia sáu màn là khoảng 1,25M/màn, trước khi tính phân bổ thích nghi; đây mới chỉ tương đương quy mô hai màn dễ đã biết. Skill có thể giúp tận dụng bước, nhưng chưa có hệ số tiết kiệm đo được. **4–5 màn trên 90% là mục tiêu lập kế hoạch hợp lý, 6 là mục tiêu mở rộng; chưa có căn cứ hứa 8 hoặc 32.** Mốc này là ước lượng, không phải kết quả dự báo đã được hiệu chuẩn.

Mọi nhánh thăm dò có cập nhật neural đều phải tính vào ngân sách nếu trần 10M áp dụng cho toàn chiến dịch. Ví dụ hai nhánh 262.144 bước tốn 524.288 bước; khi giữ một nhánh, không thể coi nhánh bị bỏ là miễn phí. Khi quay về checkpoint cũ, không đặt lại lượng bước đã tiêu. Chẩn đoán cơ học và đọc log hiện tại không dùng ngân sách huấn luyện đó.

## 11 Các phép thử còn cần trước khi cam kết cả lượt

| Thứ tự | Phép thử | Kết luận mà nó có thể hỗ trợ |
|---|---|---|
| 1 | Restore cùng snapshot và lịch sử ảnh, chạy cùng hành động vài bước | Khởi tạo đoạn đúng; sai ở đây thì chưa có ý nghĩa đo curriculum. |
| 2 | Cùng mạng và cùng số bước: toàn màn so với trộn 75% toàn màn, 25% đoạn đa dạng | Luyện đoạn có tăng số cờ và kỹ năng ở tình huống chưa luyện không. |
| 3 | Gamma 0,90 so với 0,97, giữ lambda và mọi lựa chọn khác | Có lợi cho chuỗi nhảy hay chỉ làm critic nhiễu hơn. |
| 4 | Nature so với residual nhỏ, cùng curriculum và ngân sách | Có tăng hiệu quả theo bước đủ để chấp nhận chi phí CPU không. |
| 5 | Toàn ảnh so với thêm cách gộp theo vùng | Giá trị của chia vùng độc lập với thay cả backbone. |

Không đủ ngân sách để chạy tất cả các cặp đủ dài và nhiều seed rồi vẫn dành trọn 10M cho lượt chính. Chỉ chọn **một câu hỏi học** quan trọng nhất trước; ưu tiên số 2 nếu snapshot được dùng. Các pilot 200–300k bước chỉ phát hiện lỗi lớn hoặc tín hiệu ban đầu, thường chưa đủ kết luận cấu hình nào sẽ vượt 90% ở cuối.

Nếu trần 10M bao gồm cả mọi nhánh nghiên cứu và cần dồn gần hết cho một mạng, chọn cấu hình trước, theo dõi có cổng quyết định, và ghi rõ giới hạn của một seed. Chưa có thí nghiệm huấn luyện mới trong báo cáo này để tuyên bố curriculum, residual, gamma hay chia vùng đã thắng đối chứng.

## 12 Những điểm cần điều chỉnh trong bigidea

Trong [claudebigidea](claudebigidea.md), kết luận “luyện đoạn không tạo kỹ năng mang sang màn khác” đi xa hơn dữ liệu. Phép thử đang có đo hành vi mạng cũ từ một số trạng thái, chưa có đối chứng huấn luyện skill rồi kiểm tra chuyển giao. Cách diễn đạt đúng là **chưa chứng minh được chuyển giao**.

Các con số Procgen về số màn và thời lượng huấn luyện mô tả benchmark đó; không tạo ra giới hạn toán học rằng SMB cần đúng quy mô ấy. Có thể dùng chúng để cảnh báo rủi ro tổng quát hóa, không dùng để bác bỏ mọi hướng học skill trong 10M.

Hai bản [codexbigidea](../codexbigidea.md) và claudebigidea có các phương án nối tiếp mạng cũ. Theo lựa chọn mới, chúng chỉ là bằng chứng tham khảo; lượt chính bắt đầu từ trọng số ngẫu nhiên. Tốc độ tiến bộ từ pilot nối tiếp không được coi là tốc độ của mạng mới.

Việc lấy mẫu nhiều màn trong mỗi update vẫn có cơ sở mạnh từ số đo chống quên của nhóm. Bài học mới cần bổ sung tiêu chí kỹ năng và trạng thái đa dạng, thay vì chỉ quyết định màn nào nhận thêm môi trường.

Một hướng tăng độ bền có thể xét sau là regularization với ảnh biến đổi nhẹ. Phải biến đổi nhất quán cả bốn frame, giữ thông tin mép hố và hướng hành động. Với PPO không nên tự tiện thay ảnh trong buffer rồi giữ `old_logp` của ảnh gốc; DrAC đưa các ảnh biến đổi vào loss nhất quán phụ để tránh làm sai tỉ số chính sách. [DrAC](https://arxiv.org/html/2006.12862v2)

**Hướng chốt để triển khai tiếp:** một PPO mới từ đầu, policy chung nhận bốn ảnh, nhiều màn xuất hiện sớm, bài tập nhảy đa dạng gắn với tiếp đất và chuỗi hành động, phân bổ bước theo tiến bộ và quên, learning rate còn đủ lớn khi học mới rồi giảm ở cuối. Giữ Nature-CNN làm lựa chọn mặc định trên CPU hiện tại; residual và chia vùng là các thay đổi có đối chứng, thay vì mặc định chúng sẽ đem lại chuyển giao.
