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
- Ngày 04/11/2026: B hoàn tất `core_sgd_s1` và `core_sam_s1`, bắt đầu `rho` ablation trên branch `run/rho-s1` bằng notebook stage 2 `sam_stage2_reproduction_vi_v1`.
- Ngày 06/11/2026: tám core runs đã chạy xong, chờ review chéo và tổng hợp. `core_sgd_s0` chạy trên T4, `core_sam_s0` chạy trên T4 rồi chuyển sang G4; các seed còn lại chạy trên G4.
- Ngày 08/11/2026: stage 2 đã chạy xong. Thêm notebook `sam_stage2_resnet32_noise_vi_v1` cho label noise với ResNet-32 theo thiết lập của paper, bắt đầu với seed 2 trên `run/rn32-noise40-s2` và `run/rn32-noise80-s2`.

Kế hoạch chi tiết, phân công và tiêu chí hoàn thành nằm trong [`docs/2026-10-23-sam-experiment-work-plan-vi.md`](docs/2026-10-23-sam-experiment-work-plan-vi.md). Tổng quan mục tiêu stage 1, stage 2 và các quyết định còn mở nằm trong [`docs/2026-11-06-stage1-stage2-overview-vi.md`](docs/2026-11-06-stage1-stage2-overview-vi.md).

## Branch và core runs

`main` chứa notebook chuẩn với `RUN_PLAN = "smoke"`. Mỗi branch chạy hai core runs và chỉ khác `main` ở cell cấu hình của notebook.

| Branch | Thành viên | `SEED` | Run IDs |
|---|---|---:|---|
| `run/core-s0` | A | 0 | `core_sgd_s0`, `core_sam_s0` |
| `run/core-s1` | B | 1 | `core_sgd_s1`, `core_sam_s1` |
| `run/core-s2` | C | 2 | `core_sgd_s2`, `core_sam_s2` |
| `run/core-s3` | D | 3 | `core_sgd_s3`, `core_sam_s3` |

Mỗi branch dùng chung recipe: WRN-28-10, CIFAR-10 với 45.000 ảnh train và 5.000 ảnh validation, 200 epochs, batch size 128, learning rate 0.1 với cosine schedule, momentum 0.9, weight decay 5e-4 và `rho = 0.05` cho SAM.

## Stage 2: `rho` ablation

Stage 2 dùng notebook riêng `notebooks/sam_stage2_reproduction_vi.ipynb`, nhân bản từ notebook core `sam_reproduction_vi_v5`, để không phải đổi `NOTEBOOK_VERSION` của notebook core khi các core run khác còn đang chạy. Data, model, training loop và artifact giữ nguyên; mỗi lần **Run all** chạy SAM với `rho = 0.01`, `0.10` rồi `0.20`. Điểm `rho = 0.05` lấy từ `core_sam_s1`.

| Branch | Thành viên | `SEED` | Run IDs |
|---|---|---:|---|
| `run/rho-s1` | B | 1 | `rho_001_s1`, `rho_010_s1`, `rho_020_s1` |

Cách chạy và resume giống core run (các mục bên dưới), chỉ khác:

- Tải `notebooks/sam_stage2_reproduction_vi.ipynb` từ branch `run/rho-s1`.
- Output của cell cấu hình phải có `run_plan=rho`, `seed=1`, `runs=['rho_001_s1', 'rho_010_s1', 'rho_020_s1']` và `notebook_version=sam_stage2_reproduction_vi_v1`.
- Mỗi run tốn thời gian GPU và dung lượng Drive tương đương `core_sam_s1`; ba run cần thêm khoảng 3,4 GB trên Drive.

## Stage 2: SGD 400 epochs và label noise

Ba extension còn lại dùng notebook riêng `notebooks/sam_stage2_extensions_vi.ipynb`, cũng nhân bản từ `sam_reproduction_vi_v5`. Training code giữ nguyên; `RUN_PLAN` chọn method được chạy, tỷ lệ label noise và số epoch. SAM dùng `rho = 0.05` ở mọi plan, kể cả label noise (nhóm chốt ngày 06/11/2026).

| Branch | Thành viên | `RUN_PLAN` | `SEED` | Run IDs | Drive cần thêm |
|---|---|---|---:|---|---:|
| `run/sgd400-s0` | A | `sgd400` | 0 | `compute_sgd400_s0` | ~1,1 GB |
| `run/noise40-s2` | C | `noise40` | 2 | `noise40_sgd_s2`, `noise40_sam_s2` | ~2,3 GB |
| `run/noise80-s2` | D | `noise80` | 2 | `noise80_sgd_s2`, `noise80_sam_s2` | ~2,3 GB |

Cách chạy và resume giống core run (các mục bên dưới), chỉ khác:

- Tải `notebooks/sam_stage2_extensions_vi.ipynb` từ branch được giao.
- Output của cell cấu hình phải có đúng `run_plan`, `seed` và `runs` theo bảng trên, cùng `notebook_version=sam_stage2_extensions_vi_v1 (pipeline từ sam_reproduction_vi_v5)`.
- `compute_sgd400_s0` tốn thời gian GPU khoảng bằng một core SAM run. Nên chạy toàn bộ trên G4 và kiểm tra cột `gpu_name`; số liệu thời gian chỉ so được với các epoch chạy trên cùng loại GPU.
- Label-noise runs dùng seed 2 cho cả C và D để đặt cạnh `core_sgd_s2` và `core_sam_s2` (0% noise).

## Stage 2: label noise với ResNet-32 theo paper

Notebook `notebooks/sam_stage2_resnet32_noise_vi.ipynb` chạy lại thí nghiệm label noise theo đúng model và `rho` của paper (Section 3.3, Table 4): ResNet-32 (He et al., 2016, khoảng 0,46M tham số), 200 epochs, `rho = 0.1` ở 40% noise và `rho = 0.05` ở 80% noise. Notebook nhân bản từ `sam_stage2_extensions_vi_v1`; chỉ cell model có thêm `resnet32`, còn data, label noise, training loop, SAM, checkpoint và artifact giữ nguyên. Paper không ghi batch size, learning rate, weight decay và augmentation cho thí nghiệm này, nên notebook dùng recipe chung của nhóm.

| Branch | `RUN_PLAN` | `SEED` | Run IDs |
|---|---|---:|---|
| `run/rn32-noise40-s2` | `rn32_noise40` | 2 | `rn32_noise40_sgd_s2`, `rn32_noise40_sam010_s2` |
| `run/rn32-noise80-s2` | `rn32_noise80` | 2 | `rn32_noise80_sgd_s2`, `rn32_noise80_sam005_s2` |

Cách chạy và resume giống core run (các mục bên dưới), chỉ khác:

- Tải `notebooks/sam_stage2_resnet32_noise_vi.ipynb` từ branch được giao. Hai branch độc lập, có thể chạy song song trên hai phiên Colab.
- Output của cell cấu hình phải có `model=resnet32`, đúng `run_plan`, `seed`, `rho` và `runs` theo bảng trên, cùng `notebook_version=sam_stage2_resnet32_noise_vi_v1 (pipeline từ sam_reproduction_vi_v5)`.
- Seed 2 nhận đúng tập nhãn nhiễu mà `noise40_*_s2` và `noise80_*_s2` (WRN-28-10) đã dùng. Nếu thời gian cho phép, mở thêm seed 0, 1, 3 bằng branch `run/rn32-noise40-s<k>` và `run/rn32-noise80-s<k>`.
- Mỗi checkpoint ResNet-32 chỉ vài MB. Ghi lại `epoch_seconds` và `gpu_name` của seed 2 để ước lượng thời gian trước khi mở thêm seed.
- Notebook chọn checkpoint bằng validation nhãn sạch. Paper không ghi checkpoint dùng để báo cáo; khi đặt số cạnh Table 4, báo cáo thêm test accuracy tại `last_checkpoint.pt`.

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
│   ├── 2026-10-23-sam-experiment-work-plan-vi.md
│   ├── 2026-11-06-core-s0-mixed-gpu-comparability-vi.md
│   └── 2026-11-06-stage1-stage2-overview-vi.md
├── notebooks/
│   ├── sam_reproduction_vi.ipynb          # core matrix
│   ├── sam_stage2_reproduction_vi.ipynb   # stage 2: rho ablation
│   ├── sam_stage2_extensions_vi.ipynb     # stage 2: SGD 400 epochs và label noise
│   └── sam_stage2_resnet32_noise_vi.ipynb # stage 2: label noise với ResNet-32 theo paper
└── tools/
    └── run_notebook_locally.py  # chạy thử notebook trên máy local với Drive giả lập
```

## Tài liệu tham khảo chính

- Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur. *Sharpness-Aware Minimization for Efficiently Improving Generalization*. ICLR 2021. [OpenReview](https://openreview.net/forum?id=6Tm1mposlrM)
- Mã nguồn chính thức: [google-research/sam](https://github.com/google-research/sam)
