- **Bước 0:** Tái tạo file checkpoints
  - Giải nén các file zip trong thư mục `checkpoints/` để tạo file `best.ckpt`
- **Bước 1:** Rút trích 4 loại đặc trưng từ video
  - Sử dụng [link Google Colab](https://colab.research.google.com/drive/1ML7sVxsNlMsqnkxnV6HjdLcwihrRJ_P8#scrollTo=qu4nMmfvimX-)
  - Upload video và chèn link video tương ứng vào code
  - Chạy code để rút trích đặc trưng
  - Download 4 file đặc trưng đã rút trích về máy và đặc vào thư mục `features/`
- **Bước 2:** Chạy code để inference
  - Chạy file `inference.py`
  ```bash
  python inference.py
  ```

# Note

```
- Thứ 3 => Tối 20h
- Thứ 6 => Tối 20h
```

- Cố gắng chọn video có kết quả tốt để demo: 3-5
- Hỏi thầy có cần làm slide không?
  - Vì trong file PDF không có yêu cầu này
- Rút trích 2 đặc trưng cuối cùng
- Viết code để inference cho 1 video
- Có khả năng tuần sau sẽ có quiz
