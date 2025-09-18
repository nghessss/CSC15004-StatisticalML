## Slide 2: Video Captioning

- Đầu tiên, chúng ta sẽ tìm hiểu về **bài toán Video Captioning**

## Slide 3: Video Captioning

- **Đầu vào** của mô hình là **một đoạn video**, và **đầu ra** là **một câu mô tả** các _sự kiện, hành đồng và đối tượng trong video_ đó

## Slide 4: Kiến trúc BTKG

- Mô hình được nhóm lựa chọn là **Bidirectional Transformer with Knowledge Graph**, gồm **2 encoder** và **2 decoder**

## Slide 5: Spatio-Temporal Encoder - STE

- Đầu tiên là **Spatio-Temporal Encoder**, tập trung vào việc nắm bắt các **đặc trưng hình ảnh và chuyển động chi tiết** trong video

- Đầu vào bao gồm:

  - **Đặc trưng hình ảnh** được **trích xuất từ các khung hình** bằng mô hình **Inception-ResNet-V2**
  - **Đặc trưng chuyển động** được **trích xuất từ các đoạn video ngắn** bằng mô hình **I3D**

## Slide 6: Spatio-Temporal Encoder - STE

- Đặc biệt, ta sẽ dùng **attention với 128 head**, giúp mô hình **nắm bắt được các chuyển động và tương tác chi tiết trong video**

- 2 đặc trưng này **cuối cùng được cộng lại** và cung cấp cho **Backward Decoder**

## Slide 7: Objects and Relationships Encoder - ORE

- Tiếp theo là **Objects and Relationships Encoder**, giúp xây dựng 1 **biểu diễn toàn diện và giàu ngữ nghĩa** về nội dung video

- Đầu vào gồm 4 loại đặc trưng:

  - 2 đặc trưng đầu tiên **tương tự như STE**

  - Đối với đặc trưng thứ 3, ta dùng Mask R-CNN để **xác định các đối tượng trong video, cùng với vị trị và đặc trưng của chúng**

  - Sau đó, ta dùng **đồ thị tri thức TransE** để **suy ra mối quan hệ** giữa các đối tượng này

## Slide 8: Objects and Relationships Encoder - ORE

- Tùy vào bản chất dữ liệu mà ta áp dụng các bước xử lý khác nhau:

  - **Không dùng Positional Encoding** cho **2 đặc trưng cuối cùng**
  - **Không dùng attention** cho **đặc trưng về quan hệ**

- Cuối cùng, ta **cộng tất cả đặc trưng lại** và cung cấp cho **Forward Decoder**

## Slide 9: Backward Decoder

- Backward Decoder **được huấn luyện để tạo caption theo chiều ngược** (từ cuối câu đến đầu câu)

- Thông qua đó, nó **học cách tạo ra một vector bối cảnh tóm tắt thông tin trong tương lai** và cung cấp cho **Forward Decoder**

## Slide 10: Forward Decoder

- Forward Decoder **tổng hợp tất cả thông tin từ ORE và Backward Decoder** để **tạo ra caption thực sự cho video**

- **"Biểu diễn toàn diện" từ ORE** giúp mô hình **hiểu được "video đang nói về cái gì"**

- Đồng thời, **"vector bối cảnh chứa thông tin tương lai" từ Backward Decoder** giúp mô hình chọn ra **những từ phù hợp để tạo câu văn hợp lý và mạch lạc**

## Slide 11: Loss function

- BTKG sử dụng **1 hàm lỗi tổng hợp**, kết hợp **từ 2 hàm lỗi cross-entropy của 2 decoder**

- Nhóm tác giả chọn siêu tham số **λ = 0.6** để mô hình đạt hiệu suất tốt nhất

## Slide 12: CÁC CẢI TIẾN CỦA NHÓM

- Nhóm đã **thực hiện một số cải tiến để nâng cao hiệu suất** của mô hình BTKG

## Slide 13: Lightweight MLP-Fusion Module

- Đầu tiên là ở việc **kết hợp đặc trưng của các encoder**

- Trong bài báo gốc, tác giả sử dụng **phép cộng element-wise** để kết hợp các đặc trưng

- Ở đây, nhóm sẽ "ghép nối" các đặc trưng này và sử dụng **một mô-đun tuyến tính để kết hợp chúng lại với nhau**

- Trong quá trình huấn luyện, mô-đun này sẽ **được điều chỉnh trọng số để tối ưu hóa việc kết hợp đặc trưng**

## Slide 14: ResiDual: Transformer with Dual Residual Connections

- Tiếp theo, nhóm đã **thay đổi kiến trúc Transformer** của mô hình

- 2 kiến trúc phổ biến là **Post-LayerNorm** và **Pre-LayerNorm** đều có những nhược điểm

  - **Post-LayerNorm:** dễ bị **gradient vanishing** khi mô hình quá sâu
  - Còn với **Pre-LayerNorm:**, các nhà nghiên cứu đã chỉ ra rằng các **biểu diễn ẩn ở các tầng sâu có xu hướng trở nên giống nhau**, làm **giảm năng lực thổng thể của mô hình**

- Do đó, nhóm đã sử dụng kiến trúc **ResiDual** để **tận dụng những gì tốt nhất của cả 2 bên**

- Residual **sử dụng 2 kết nối**, bên trái là Post-LayerNorm, bên phải là Pre-LayerNorm

## Slide 15: Thực nghiệm của nhóm

- Đến với phần thực nghiệm của mô hình video captioning

## Slide 16: Thiết lập thực nghiệm

- Nhóm sử dụng tập dữ liệu MSVD gồm **1.970 video ngắn**, chia thành **3 tập train, valid và test** theo **tỷ lệ tiêu chuẩn**

## Slide 17: Thiết lập thực nghiệm

- Nhóm đã **chạy lại mô hình gốc** thì thấy **kết quả thấp hơn ở tất cả chỉ số** so với số liệu trong paper, điều này có thể **bắt nguồn từ nhiều yếu tố** như **phần cứng**, **phiên bản của thư viện**, v.v.

- Nhìn chung, các cải tiến của nhóm đều cho thấy **sự cải thiện đáng kể về hiệu suất** ở tất cả chỉ số so với mô hình gốc, đặc biệt là **khi thay đổi sang kiến trúc ResiDual**

## Slide 18: IMAGE captioning

- Tiếp theo là mô hình IMAGE captioning
