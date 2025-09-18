## Cải tiến so với mô hình gốc

### Cải tiến phương pháp kết hợp đặc trưng đa phương thức

Trong kiến trúc BTKG gốc, các đặc trưng đa phương thức (multimodal features) được kết hợp lại bằng cách sử dụng phép cộng theo từng phần tử (element-wise addition). Cụ thể, trong Bộ mã hóa Không gian-Thời gian (Spatio-Temporal Encoder - STE), đặc trưng hình ảnh và đặc trưng chuyển động được tổng hợp qua công thức:

$$E_{STE} = E_I \oplus E_M$$

Tương tự, trong Bộ mã hóa Đối tượng và Quan hệ (Objects and Relationships Encoder - ORE), cả bốn loại đặc trưng được kết hợp bằng công thức

$$E_{ORE} = E_I \oplus E_M \oplus E_o \oplus E_r$$

Phương pháp cộng trực tiếp này có một hạn chế cố hữu: nó giả định rằng mỗi phương thức (modality) đóng góp một vai trò quan trọng như nhau vào biểu diễn cuối cùng. Đây là một phép toán tĩnh, không có khả năng học và không thể tự động điều chỉnh trọng số của từng loại đặc trưng dựa trên ngữ cảnh của video.

Để khắc phục nhược điểm này, nhóm chúng tôi đã đề xuất một cải tiến bằng cách thay thế phép cộng theo từng phần tử bằng một mô-đun **Feature Fusion** có khả năng học. Mô-đun này sử dụng một layer tích chập một chiều (`Conv1D`) với kích thước kernel bằng 1, hoạt động như một tầng tuyến tính (linear layer) học được để kết hợp các đặc trưng một cách linh hoạt.

#### Kiến trúc mô-đun Feature Fusion

Mô-đun `FeatureFusion` nhận vào một danh sách các tensor đặc trưng và thực hiện các bước sau:

1.  **Ghép nối (Concatenation):** Thay vì cộng, các tensor đặc trưng từ các nguồn khác nhau (ví dụ: $E_I, E_M, E_o, E_r$) có cùng shape `(batch_size, seq_len, d_model)` được ghép nối lại với nhau theo chiều cuối cùng (chiều đặc trưng). Thao tác này tạo ra một tensor mới có shape `(batch_size, seq_len, n_features * d_model)`, trong đó `n_features` là số lượng nguồn đặc trưng.

2.  **Tích chập 1x1 (1x1 Convolution):** Tensor kết hợp sau đó được đưa qua một layer `Conv1D` với `kernel_size=1`. Tầng tích chập này có tác dụng như một phép chiếu tuyến tính (linear projection) học được, ánh xạ không gian đặc trưng từ `n_features * d_model` trở về `d_model`. Về bản chất, nó học cách "trộn" các đặc trưng đã ghép nối để tạo ra một biểu diễn mới, súc tích hơn.

3.  **Định hình lại đầu ra:** Tensor đầu ra từ layer `Conv1D` có shape `(batch_size, seq_len, d_model)`, hoàn toàn tương thích với phần còn lại của kiến trúc Transformer và có thể thay thế trực tiếp cho kết quả của phép cộng element-wise trước đây.

#### Ưu điểm của cải tiến

Việc thay thế này mang lại những lợi ích đáng kể:

- **Khả năng học (Learnability):** Đây là ưu điểm lớn nhất. Các trọng số của layer `Conv1D` được cập nhật trong quá trình huấn luyện. Điều này cho phép mô hình tự động **học được tầm quan trọng tương đối** của từng loại đặc trưng. Ví dụ, đối với một video có nhiều hành động phức tạp, mô hình có thể học cách gán trọng số cao hơn cho đặc trưng chuyển động ($E_M$) và đặc trưng quan hệ ($E_r$).

- **Tăng cường khả năng biểu diễn:** Bằng cách cho phép các đặc trưng tương tác với nhau thông qua một phép biến đổi tuyến tính, mô hình có thể tạo ra một không gian biểu diễn chung (joint representation space) phong phú và có ý nghĩa hơn so với việc chỉ cộng chúng lại một cách đơn giản.

- **Hiệu quả tính toán:** Một layer tích chập 1x1 là một phép toán rất hiệu quả, không làm tăng đáng kể chi phí tính toán hay độ phức tạp của mô hình nhưng lại cải thiện đáng kể sức mạnh biểu diễn.

### Cải tiến Kiến trúc Transformer với Kết nối Thặng dư Kép (ResiDual)

#### Bối cảnh và Vấn đề

Kiến trúc Transformer trong mô hình BTKG gốc sử dụng cơ chế **Post-Layer Normalization (Post-LN)**. Trong cấu trúc này, kết nối thặng dư (residual connection) được thực hiện bằng cách cộng đầu vào và đầu ra của một khối con (ví dụ: Multi-Head Attention), sau đó mới áp dụng phép Chuẩn hóa layer (Layer Normalization). Sơ đồ của Post-LN được minh họa trong Hình (a) dưới đây.

Mặc dù phổ biến, kiến trúc Post-LN tồn tại những hạn chế nghiêm trọng, đặc biệt khi xây dựng các mô hình sâu:

1.  **Vấn đề Triệt tiêu Gradient (Gradient Vanishing):** Tín hiệu gradient khi lan truyền ngược từ các layer trên xuống các layer dưới sẽ bị suy yếu theo cấp số nhân. Điều này là do gradient phải đi qua nhiều lần chuẩn hóa layer, khiến cho các layer gần đầu vào hầu như không được cập nhật, làm cho việc huấn luyện các mô hình sâu trở nên rất khó khăn và không ổn định.

2.  **Huấn luyện không ổn định:** Để huấn luyện thành công các mô hình Transformer sâu sử dụng Post-LN, người ta thường phải dùng đến các kỹ thuật bổ trợ như "learning-rate warm-up" (tăng dần tốc độ học trong các bước đầu) để tránh sự bất ổn định.

#### Giải pháp đề xuất: Áp dụng kiến trúc ResiDual

Để giải quyết các vấn đề trên và tăng cường sự ổn định cũng như hiệu suất của mô hình, nhóm chúng tôi đã thay thế kiến trúc Post-LN gốc bằng **ResiDual**, một kiến trúc tiên tiến được đề xuất bởi Xie và các cộng sự tại Microsoft. ResiDual được thiết kế để kết hợp những ưu điểm của cả hai phương pháp Post-LN và Pre-LN (Pre-Layer Normalization), đồng thời loại bỏ các nhược điểm của chúng.

Kiến trúc ResiDual (minh họa trong Hình (c) ở trên) sử dụng **hai nhánh kết nối thặng dư song song**:

- **Nhánh chính (tương tự Post-LN):** Nhánh này giữ lại cấu trúc của Post-LN, nơi đầu ra của khối con được cộng với đầu vào và sau đó được chuẩn hóa. Nhánh này có vai trò quan trọng trong việc **duy trì sự đa dạng của các biểu diễn** (representation diversity), giúp chống lại hiện tượng "sụp đổ biểu diễn" (representation collapse) thường thấy ở Pre-LN.

- **Nhánh kép (dual branch, tương tự Pre-LN):** Một nhánh thặng dư thứ hai được thêm vào, cho phép tín hiệu (cả xuôi và ngược) đi tắt qua các khối. Nhánh này tích lũy đầu ra của các khối con và cho phép gradient lan truyền trực tiếp về các layer sâu nhất mà không bị suy yếu bởi các layer chuẩn hóa. Điều này giúp **giải quyết triệt để vấn đề triệt tiêu gradient**.

#### Lợi ích của việc áp dụng ResiDual

Việc tích hợp ResiDual vào các khối Transformer của mô hình BTKG mang lại những ưu điểm vượt trội:

- **Ổn định Huấn luyện và Hội tụ Tốt hơn:** Bằng cách cung cấp một đường dẫn thông suốt cho gradient, ResiDual giúp việc huấn luyện trở nên ổn định hơn rất nhiều, đặc biệt khi tăng độ sâu của mô hình. Nó làm giảm sự phụ thuộc vào các kỹ thuật như learning-rate warm-up, giúp quá trình huấn luyện đơn giản và hiệu quả hơn.

- **Tăng cường Năng lực Biểu diễn:** Kiến trúc này ngăn chặn hiện tượng "sụp đổ biểu diễn", đảm bảo rằng các layer sâu hơn vẫn có thể học và tinh chỉnh các đặc trưng một cách hiệu quả. Điều này cho phép mô hình tận dụng tối đa năng lực của các tham số, dẫn đến một mô hình mạnh mẽ hơn.

## Đánh giá mô hình

Bảng dưới đây trình bày kết quả so sánh giữa phiên bản được công bố trong bài báo gốc, kết quả tái triển khai mô hình gốc của nhóm, và kết quả của các phiên bản với những cải tiến của chúng tôi.

| Phiên bản mô hình                | BLEU@4 | METEOR | ROUGE-L | CIDEr-D |
| -------------------------------- | ------ | ------ | ------- | ------- |
| Kết quả được công bố             | 55.7   | 38.3   | 74.7    | 104.5   |
| BTKG (gốc)                       | 54.6   | 37.9   | 74.3    | 96.8    |
| BTKG + Feature Fusion            | 55.1   | 38.0   | 74.1    | 98.1    |
| BTKG + Feature Fusion + ResiDual | 55.5   | 38.3   | 74.9    | 100.2   |

**So sánh với kết quả gốc:** Phiên bản BTKG gốc do nhóm tái triển khai có điểm số thấp hơn một chút so với kết quả được công bố trong bài báo. Cụ thể, điểm CIDEr-D giảm từ 104.5 xuống 96.8. Sự khác biệt này có thể đến từ nhiều yếu tố, bao gồm sự khác biệt về môi trường thực nghiệm (bài báo gốc sử dụng GPU RTX3060, trong khi nhóm sử dụng NVIDIA Tesla P100 trên Kaggle), hoặc sự khác biệt nhỏ trong quá trình tiền xử lý dữ liệu. Tuy nhiên, kết quả này vẫn là một đường cơ sở (baseline) vững chắc để đánh giá tác động của các cải tiến mà nhóm đề xuất.

**Hiệu quả của Module Feature Fusion:** Khi thay thế phép cộng element-wise bằng module Feature Fusion, mô hình đã cho thấy sự cải thiện nhẹ ở các chỉ số BLEU@4, METEOR, và đặc biệt là CIDEr-D (tăng từ 96.8 lên 98.1). Phép cộng element-wise mặc định rằng các kênh đặc trưng từ những modal khác nhau (không gian, thời gian, đối tượng, quan hệ) có tầm quan trọng như nhau. Ngược lại, module FeatureFusion sử dụng một layer tích chập 1x1 hoạt động như một cơ chế kết hợp có trọng số và có thể học được. Điều này cho phép mô hình tự động điều chỉnh và ưu tiên các luồng thông tin quan trọng hơn, từ đó tạo ra một vector đặc trưng hợp nhất (fused feature) giàu thông tin hơn để đưa vào bộ giải mã.

**Tác động tích cực của kiến trúc ResiDual:** Cải tiến đáng chú ý nhất đến từ việc tích hợp kiến trúc ResiDual vào các khối Transformer, thay thế cho cơ chế Post-Layer Normalization (Post-LN) mặc định. Phiên bản "BTKG + Feature Fusion + ResiDual" đã cải thiện đáng kể trên tất cả các chỉ số so với baseline của nhóm, đặc biệt là sự gia tăng mạnh mẽ của CIDEr-D từ 96.8 lên 100.2 và METEOR đạt ngang bằng với kết quả công bố (38.3). Việc áp dụng ResiDual giúp ổn định quá trình huấn luyện và cho phép các tầng sâu hơn của mô hình học được các biểu diễn phức tạp và đa dạng hơn. Điều này trực tiếp nâng cao khả năng của mô hình trong việc nắm bắt các ngữ nghĩa tinh vi của video, dẫn đến việc tạo ra các câu mô tả chính xác và phù hợp hơn về mặt ngữ nghĩa, được phản ánh rõ nét qua việc tăng điểm CIDEr-D - một chỉ số đo lường sự tương đồng về mặt ngữ nghĩa dựa trên sự đồng thuận của con người.

**Kết luận:** Mặc dù mô hình cải tiến tốt nhất của nhóm vẫn chưa vượt qua hoàn toàn các con số được công bố trên bài báo gốc, các kết quả thực nghiệm đã chứng minh một cách rõ ràng rằng:

- Sử dụng một cơ chế hợp nhất đặc trưng có thể học được (Feature Fusion) hiệu quả hơn so với phép cộng trực tiếp.

- Việc thay đổi kiến trúc Transformer sang một cơ chế ổn định hơn như ResiDual mang lại những cải thiện đáng kể về hiệu suất, đặc biệt là về chất lượng ngữ nghĩa của câu chữ được tạo ra.

Những kết quả này đã khẳng định hướng đi của nhóm là đúng đắn và các cải tiến được đề xuất có tiềm năng lớn trong việc nâng cao hiệu quả của mô hình BTKG.
