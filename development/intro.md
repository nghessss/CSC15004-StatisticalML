# **CHƯƠNG 1: GIỚI THIỆU**

## **1.1. Bối cảnh và Lý do chọn đề tài**

Trong kỷ nguyên số hiện nay, chúng ta đang chứng kiến sự bùng nổ của dữ liệu đa phương tiện, đặc biệt là video. Từ các nền tảng mạng xã hội như YouTube, TikTok, Facebook đến các hệ thống giám sát an ninh, khoá học trực tuyến và kho lưu trữ cá nhân, video đã trở thành một phương tiện truyền tải thông tin phổ biến và hiệu quả. Tuy nhiên, khối lượng thông tin khổng lồ được chứa đựng bên trong video lại đặt ra một thách thức lớn: làm thế nào để khai thác và truy xuất thông tin một cách nhanh chóng và chính xác?

Các phương pháp tìm kiếm truyền thống thường chỉ dựa vào siêu dữ liệu (metadata) do con người tạo ra như tiêu đề, mô tả, hoặc các thẻ (tags). Cách tiếp cận này tỏ ra hạn chế khi người dùng muốn tìm kiếm những chi tiết cụ thể bên trong nội dung của video, chẳng hạn như một hành động nhất định, một đối tượng xuất hiện trong vài giây, màu sắc của một vật thể, hay một câu nói được phát ra tại một thời điểm cụ thể. Việc xem toàn bộ video để tìm kiếm một phân đoạn thông tin mong muốn là cực kỳ tốn thời gian và không hiệu quả.

Để giải quyết bài toán này, các công nghệ Trí tuệ Nhân tạo (AI) trong lĩnh vực Thị giác Máy tính (Computer Vision) và Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing) đã mang lại những giải pháp đột phá. Các mô hình có khả năng "hiểu" được nội dung hình ảnh và video (Image/Video Captioning) cũng như "nghe" được âm thanh (Speech Recognition) đang ngày càng trở nên mạnh mẽ và chính xác.

Xuất phát từ nhu cầu thực tiễn đó, đồ án này đề xuất xây dựng một hệ thống **Chatbot thông minh cho phép người dùng hỏi đáp trực tiếp về nội dung của một video**. Thay vì phải xem toàn bộ, người dùng có thể tương tác với hệ thống thông qua ngôn ngữ tự nhiên, đặt những câu hỏi và nhận về câu trả lời chính xác dựa trên cả nội dung hình ảnh và âm thanh của video.

## **1.2. Mục tiêu của đồ án**

Đồ án tập trung vào việc giải quyết bài toán truy xuất thông tin chi tiết trong video bằng cách xây dựng và tích hợp một hệ thống chatbot thông minh với các mục tiêu cụ thể như sau:

1.  **Xây dựng bối cảnh (context) toàn diện cho video:** Tự động trích xuất và tổng hợp thông tin từ ba nguồn chính:

    - **Nội dung hình ảnh tĩnh (Image Captioning):** Tóm tắt bối cảnh chung, các đối tượng chính và thuộc tính của chúng trong các khung hình tiêu biểu của video.
    - **Nội dung hành động (Video Captioning):** Mô tả các hành động, sự kiện đang diễn ra trong video bằng cách sử dụng mô hình chuyên biệt (BTKG) để nắm bắt được yếu tố chuyển động.
    - **Nội dung âm thanh (Speech Recognition):** Chuyển đổi toàn bộ lời thoại, âm thanh có trong video sang dạng văn bản (transcript) bằng cách sử dụng mô hình **PhoWhisper**, một mô hình nhận dạng tiếng nói tiếng Việt có độ chính xác cao.

2.  **Phát triển Chatbot tương tác:** Xây dựng một chatbot có khả năng hiểu câu hỏi của người dùng bằng ngôn ngữ tự nhiên và sử dụng kho bối cảnh đã được tạo ra ở trên để tìm kiếm và đưa ra câu trả lời phù hợp.

3.  **Tích hợp và kiểm thử:** Tích hợp các module riêng lẻ (Image Captioning, Video Captioning, PhoWhisper, và Chatbot) thành một hệ thống hoàn chỉnh và kiểm thử hiệu quả của hệ thống thông qua các kịch bản hỏi đáp thực tế.

## **1.3. Kiến trúc và giải pháp đề xuất**

Để đạt được các mục tiêu trên, hệ thống được đề xuất sẽ bao gồm các thành phần chính, phối hợp chặt chẽ với nhau để tạo thành một luồng xử lý thông tin hoàn chỉnh.

**Luồng hoạt động của hệ thống:**
Khi người dùng tải lên một video, hệ thống sẽ xử lý song song qua ba module cốt lõi:

1.  **Module Image Captioning:** Hệ thống sẽ trích xuất một số khung hình chính (key frames) đại diện cho toàn cảnh của video. Các khung hình này sau đó được đưa vào mô hình Image Captioning để tạo ra các câu mô tả ngắn gọn. Ví dụ, từ một video về tập gym, module này có thể tạo ra mô tả: _"Một người đàn ông trong phòng tập gym có nhiều thiết bị"_. Nhiệm vụ của module này là tóm tắt toàn cảnh và các đối tượng tĩnh.

2.  **Module Video Captioning (dựa trên mô hình BTKG):** Toàn bộ video sẽ được phân tích bởi mô hình Video Captioning. Mô hình này được thiết kế để tập trung vào việc nhận dạng chuyển động và hành động. Kết quả là một hoặc nhiều câu mô tả các sự kiện chính diễn ra. Ví dụ: _"Một người đàn ông đang nâng tạ."_. Module này trả lời cho câu hỏi "Có chuyện gì đang xảy ra?".

3.  **Module Nhận dạng Tiếng nói (PhoWhisper):** Luồng âm thanh của video sẽ được tách ra và xử lý bởi mô hình PhoWhisper. Mô hình này sẽ chuyển đổi tất cả các đoạn hội thoại hoặc lời nói trong video thành một bản ghi đầy đủ (transcript). Ví dụ, nếu người trong video nói: _"Hôm nay tôi sẽ thử mức tạ 100kg"_, module này sẽ tạo ra chính xác đoạn văn bản đó.

Ba kết quả đầu ra này (mô tả cảnh tĩnh, mô tả hành động, và bản ghi âm thanh) sẽ được kết hợp lại để tạo thành một **văn bản bối cảnh (context)** duy nhất và toàn diện. Văn bản này chính là "bộ não" của Chatbot.

Khi người dùng đặt câu hỏi, Chatbot sẽ phân tích câu hỏi và tìm kiếm thông tin trong văn bản bối cảnh này để đưa ra câu trả lời chính xác nhất.

**Ví dụ minh họa:**

Xét một video một người đang đẩy tạ. Hệ thống sẽ tạo ra bối cảnh tương tự như:

- _Image Captioning:_ "Một người đàn ông mặc áo màu xanh và quần đen trong một phòng tập."
- _Video Captioning:_ "Người đàn ông đang thực hiện động tác đẩy tạ qua đầu."
- _PhoWhisper Transcript:_ "(Tiếng thở) Cố lên nào... thêm một lần nữa thôi."

Dựa trên bối cảnh này, Chatbot có thể trả lời các câu hỏi đa dạng của người dùng:

- _Người dùng hỏi:_ "Người trong video mặc áo màu gì?"
  - _Chatbot trả lời:_ "Người trong video mặc áo màu xanh." (Dữ liệu từ Image Captioning)
- _Người dùng hỏi:_ "Người đó đang làm gì?"
  - _Chatbot trả lời:_ "Người trong video đang thực hiện động tác đẩy tạ qua đầu." (Dữ liệu từ Video Captioning)
- _Người dùng hỏi:_ "Người này có nói gì không?"
  - _Chatbot trả lời:_ "Có, người này nói: 'Cố lên nào... thêm một lần nữa thôi.'" (Dữ liệu từ PhoWhisper)
