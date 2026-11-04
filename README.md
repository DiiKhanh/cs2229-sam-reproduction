# CS2229 - Tái lập Sharpness-Aware Minimization

Repository lưu tài liệu, notebook và kết quả thực nghiệm của nhóm khi tái lập một phần bài báo *Sharpness-Aware Minimization for Efficiently Improving Generalization* (ICLR 2021).

## Thông tin nhóm

- **Môn học:** CS2229.26.1.CH.01
- **Nhóm:** 6
- **Đề tài:** *Sharpness-Aware Minimization for Efficiently Improving Generalization*
- **Bài báo:** [OpenReview](https://openreview.net/forum?id=6Tm1mposlrM)

| MSSV | Họ và tên |
|---|---|
| 26848099 | Đặng Anh Khoa |
| 26848113 | Nguyễn Khánh Linh |
| 26848200 | Huỳnh Quốc Thái |
| 26848096 | Phạm Duy Khánh |

## Mục tiêu

So sánh SGD và SAM-SGD trên CIFAR-10 trong cùng điều kiện về mô hình, dữ liệu và quy trình huấn luyện. Thực nghiệm chính dùng WideResNet-28-10, 200 epochs, `rho = 0.05` và bốn seed, đồng thời ghi nhận accuracy, generalization gap, thời gian chạy và bộ nhớ GPU.

## Trạng thái

- Ngày 04/11/2026: nhóm chốt WRN-28-10 cho core matrix.
- Notebook `sam_reproduction_vi_v5` sẵn sàng cho tám core runs trên bốn branch `run/core-s0` đến `run/core-s3`.
- Chưa có core run nào hoàn tất.

Kế hoạch chi tiết, phân công và tiêu chí hoàn thành nằm trong [`docs/2026-10-23-sam-experiment-work-plan-vi.md`](docs/2026-10-23-sam-experiment-work-plan-vi.md).

## Branch và core runs

`main` chứa notebook chuẩn với `RUN_PLAN = "smoke"`. Mỗi branch chạy hai core runs và chỉ khác `main` ở cell cấu hình của notebook.

| Branch | Thành viên | `SEED` | Run IDs |
|---|---|---:|---|
| `run/core-s0` | A | 0 | `core_sgd_s0`, `core_sam_s0` |
| `run/core-s1` | B | 1 | `core_sgd_s1`, `core_sam_s1` |
| `run/core-s2` | C | 2 | `core_sgd_s2`, `core_sam_s2` |
| `run/core-s3` | D | 3 | `core_sgd_s3`, `core_sam_s3` |

Mỗi branch dùng chung recipe: WRN-28-10, CIFAR-10 với 45.000 ảnh train và 5.000 ảnh validation, 200 epochs, batch size 128, learning rate 0.1 với cosine schedule, momentum 0.9, weight decay 5e-4 và `rho = 0.05` cho SAM.

## Chuẩn bị Google Drive

Mỗi thành viên làm một lần trên Google Drive cá nhân:

```text
MyDrive/
└── CS2229-SAM/
    ├── data/
    │   └── cifar-10-python.tar.gz   # upload thủ công, giữ nguyên tên file
    └── results/                     # notebook tự tạo và ghi artifact vào đây
```

- File dataset phải đúng tên `cifar-10-python.tar.gz`. Nếu Drive tự đổi thành `cifar-10-python (1).tar.gz` do trùng tên, xóa bản cũ rồi đổi tên lại.
- Notebook kiểm tra MD5 chính thức của file. Nếu upload bị hỏng, notebook dừng trước khi train và báo cần upload lại.
- Mỗi core run cần khoảng 1,1 GB trên Drive (hai checkpoint WRN-28-10, mỗi checkpoint có một bản `.backup`). Hai run của một branch cần khoảng 2,3 GB.

## Chạy trên Google Colab

1. Mở branch được giao trên GitHub, vào `notebooks/sam_reproduction_vi.ipynb` và tải file về bằng nút **Download raw file**.
2. Mở [Google Colab](https://colab.research.google.com/), chọn **File -> Upload notebook** và upload file vừa tải.
3. Chọn **Runtime -> Change runtime type -> GPU**.
4. Chọn **Runtime -> Run all**, rồi cấp quyền Google Drive cho đúng tài khoản chứa thư mục `CS2229-SAM`.
5. Kiểm tra output của cell cấu hình: `run_plan=core`, `seed` đúng branch và `notebook_version=sam_reproduction_vi_v5`.

Notebook chạy SGD trước, sau đó SAM. Mỗi epoch in loss, accuracy, thời gian epoch và thời gian còn lại ước tính của run đang chạy. Không sửa cell nào trong notebook.

Khi Colab upload notebook, file được lưu trong `MyDrive/Colab Notebooks/`. Dùng lại chính file này cho các lần resume.

## Resume khi Colab hết phiên

Colab có thể ngắt phiên do giới hạn thời gian, hết quota GPU hoặc mất kết nối. Checkpoint được ghi lên Drive sau mỗi epoch, nên chỉ mất phần epoch đang chạy dở.

1. Mở lại notebook từ `MyDrive/Colab Notebooks/` (hoặc upload lại notebook của đúng branch).
2. Chọn **Runtime -> Change runtime type -> GPU** nếu runtime chưa có GPU.
3. Chọn **Runtime -> Run all** và cấp lại quyền Google Drive.
4. Kiểm tra log:
   - `core_sgd_s<k> đã hoàn tất; kiểm tra artifact và bỏ qua training.`: run SGD đã xong và được giữ nguyên.
   - `Resume core_sam_s<k> từ epoch N/200`: run đang tiếp tục từ epoch kế tiếp.

Lặp lại các bước trên cho đến khi bảng ở mục 9 của notebook có `artifact_status = OK` cho cả hai run.

Quy tắc khi resume:

- Không sửa cell cấu hình và không đổi branch giữa chừng. Notebook từ chối resume nếu cấu hình khác checkpoint.
- Không xóa, đổi tên hay di chuyển thư mục `results/<run_id>/`, kể cả file `.backup`.
- Không tự xóa checkpoint để giải phóng dung lượng. Nếu Drive đầy, kiểm tra dung lượng còn trống và Trash, rồi báo nhóm.
- Nếu gặp lỗi khác ngoài ngắt phiên, chụp log và báo nhóm trước khi chạy lại.
- Khi cả hai run đã xong, chọn **Runtime -> Disconnect and delete runtime** để không tiêu tốn thêm quota GPU.

Một run có thể được resume trên các loại GPU khác nhau. Cột `gpu_name` trong `history.csv` ghi GPU của từng epoch để giải thích thời gian chạy.

## Artifact của mỗi run

```text
MyDrive/CS2229-SAM/results/<run_id>/
├── config.json
├── history.csv                # mỗi dòng một epoch, kèm gpu_name
├── summary.json               # test accuracy tại best checkpoint theo validation
├── best_checkpoint.pt
├── best_checkpoint.pt.backup
├── last_checkpoint.pt
├── last_checkpoint.pt.backup
├── learning_curves.png
└── environment.txt
```

Checkpoint không được commit vào Git. Cách gom kết quả để tổng hợp sẽ được cập nhật sau khi có đủ core runs.

## Cấu trúc repository

```text
.
├── AGENTS.md                  # hướng dẫn cho AI agent làm việc trong repo
├── CLAUDE.md                  # trỏ tới AGENTS.md
├── docs/
│   ├── 2026-10-12-sam-paper-summary-vi.md
│   └── 2026-10-23-sam-experiment-work-plan-vi.md
├── notebooks/
│   └── sam_reproduction_vi.ipynb
└── tools/
    └── run_notebook_locally.py  # chạy thử notebook trên máy local với Drive giả lập
```

## Tài liệu tham khảo chính

- Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur. *Sharpness-Aware Minimization for Efficiently Improving Generalization*. ICLR 2021. [OpenReview](https://openreview.net/forum?id=6Tm1mposlrM)
- Mã nguồn chính thức: [google-research/sam](https://github.com/google-research/sam)
