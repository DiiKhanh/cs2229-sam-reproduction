# CS2229 - Tái lập Sharpness-Aware Minimization

Repository lưu tài liệu, notebook và kết quả thực nghiệm của nhóm khi tái lập một phần bài báo *Sharpness-Aware Minimization for Efficiently Improving Generalization* (ICLR 2021).

## Mục tiêu

So sánh SGD và SAM-SGD trên CIFAR-10 trong cùng điều kiện về mô hình, dữ liệu và quy trình huấn luyện. Thực nghiệm chính sử dụng nhiều seed, đồng thời ghi nhận accuracy, generalization gap, thời gian chạy và bộ nhớ GPU.

## Cấu trúc repository

```text
.
├── docs/
│   ├── 2026-10-12-sam-paper-summary-vi.md
│   └── 2026-10-23-sam-experiment-work-plan-vi.md
├── notebooks/
│   └── sam_reproduction_vi.ipynb
├── assets/      # Hình ảnh dùng trong tài liệu, tạo khi cần
└── results/     # Kết quả thực nghiệm, tạo khi có run hoàn chỉnh
```

Hai thư mục `assets/` và `results/` chưa được tạo cho đến khi có nội dung thực tế.

## Chạy notebook

1. Mở `notebooks/sam_reproduction_vi.ipynb` bằng Google Colab.
2. Chọn GPU tại **Runtime > Change runtime type**.
3. Chạy toàn bộ notebook để mount Google Drive và thực hiện smoke test cho SGD và SAM.
4. Kiểm tra artifact của smoke test trước khi chuyển sang cấu hình chạy đầy đủ.

Notebook lưu kết quả vào `MyDrive/CS2229-SAM/results` và dừng nếu Google Drive không được mount thành công. Các thành viên cần thống nhất model và phiên bản notebook trước khi chạy ma trận thực nghiệm chính.

## Tài liệu tham khảo chính

- Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur. *Sharpness-Aware Minimization for Efficiently Improving Generalization*. ICLR 2021. [OpenReview](https://openreview.net/forum?id=6Tm1mposlrM)
- Mã nguồn chính thức: [google-research/sam](https://github.com/google-research/sam)

## Trạng thái

Notebook hiện chứa output của smoke test đã lưu trước đó. Cần chạy lại notebook trong môi trường của từng thành viên để xác minh khả năng tái lập và tạo artifact tương ứng với phiên bản hiện tại.
