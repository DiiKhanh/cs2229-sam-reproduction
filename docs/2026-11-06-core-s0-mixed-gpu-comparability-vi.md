# Khả năng so sánh core runs khi seed 0 chạy trên T4 và G4

- **Ngày ghi:** 06/11/2026
- **Phạm vi:** 8 core runs của stage 1 (`core_sgd_s0` đến `core_sam_s3`), notebook `sam_reproduction_vi_v5`
- **Trạng thái:** Kết luận dựa trên đọc code notebook. Chưa đối chiếu với `history.csv` và `summary.json` thật trên Drive (xem Mục 5).

## 1. Bối cảnh

Theo báo cáo của các thành viên:

| Run | GPU |
|---|---|
| `core_sgd_s0` | T4 toàn bộ 200 epochs |
| `core_sam_s0` | T4 ở khoảng 100 epochs đầu, G4 ở khoảng 100 epochs cuối (resume sang G4) |
| `core_*_s1`, `core_*_s2`, `core_*_s3` | G4 |

Câu hỏi: kết quả của seed 0 có ghép được với seed 1-3 trong bảng tổng hợp stage 1 không?

## 2. Kết luận

| Nhóm chỉ số | So sánh được giữa các seed? | Cách xử lý |
|---|---|---|
| Test accuracy, best validation accuracy, generalization gap, train loss/accuracy | Có | Tổng hợp mean ± SD trên cả 4 seed như kế hoạch. Ghi phần cứng của seed 0 vào mục hạn chế. |
| Time/epoch, total training time, overhead SAM/SGD | Không, nếu gộp các loại GPU | Chỉ tính theo từng loại GPU, lấy từ `history.csv` (Mục 4). |
| Peak GPU memory | Gần như có, nhưng chưa chắc chắn | Ghi kèm loại GPU cạnh mỗi con số. |

## 3. Vì sao accuracy và generalization gap vẫn so sánh được

1. **Các yếu tố quyết định kết quả giống nhau.** Mọi core run dùng cùng `NOTEBOOK_VERSION = sam_reproduction_vi_v5`, cùng recipe, cùng validation split (`SPLIT_SEED = 20261023`) và cùng quy tắc chọn checkpoint theo validation accuracy. Loại GPU không thay đổi thuật toán, dữ liệu hay hyperparameter.
2. **Resume trên GPU khác không làm sai luồng ngẫu nhiên.** `capture_random_state()` lưu RNG state của Python, NumPy, PyTorch CPU, CUDA và DataLoader generator vào checkpoint. Khi resume, thứ tự mini-batch và augmentation tiếp tục đúng như khi chạy liền mạch.
3. **Khác biệt phần cứng chỉ gây sai số số học nhỏ.** Notebook không dùng AMP và không đặt cờ TF32. Theo mặc định của PyTorch, `torch.backends.cudnn.allow_tf32 = True`:
   - T4 (Turing) không hỗ trợ TF32, nên convolution chạy FP32 đầy đủ.
   - G4 là GPU đời mới hơn, có hỗ trợ TF32, nên convolution chạy TF32.

   Mức chênh này nhỏ hơn nhiều so với độ phân tán giữa các seed, nên không thay đổi kết luận SAM so với SGD.

### Hạn chế cần nêu trong báo cáo

- `seed_everything()` đặt `cudnn.deterministic = True`. Cờ này chỉ đảm bảo lặp lại kết quả trên cùng loại phần cứng và cùng phiên bản phần mềm. Nếu chạy lại `core_sam_s0` hoàn toàn trên G4, kết quả sẽ không trùng từng chữ số với run hiện tại.
- Seed 0 chạy trên phần cứng khác seed 1-3. Đây là một nguồn biến thiên mà các seed còn lại không có.

## 4. Vì sao chỉ số compute phải tách theo GPU

T4 chậm hơn G4 đáng kể. Gộp epoch của hai loại GPU sẽ làm sai cả time/epoch lẫn tỷ lệ overhead.

### 4.1 Hai trường dễ gây hiểu nhầm trong `summary.json` của `core_sam_s0`

Cả hai trường đều do `run_experiment()` ghi ra.

| Trường | Cách notebook tính | Vấn đề với `core_sam_s0` |
|---|---|---|
| `mean_epoch_seconds` | `np.mean` của `epoch_seconds` trên mọi dòng `history.csv` | Trộn epoch T4 với epoch G4 |
| `gpu_name` | Lấy từ `collect_environment()` của phiên cuối cùng | Chỉ ghi G4, dù khoảng nửa đầu run chạy trên T4 |

`total_training_seconds` cũng là tổng thời gian trên hai loại GPU. Không đưa ba trường này thẳng vào bảng compute. Hãy tính lại từ `history.csv`, nhóm theo cột `gpu_name`.

### 4.2 Cách tính overhead SAM/SGD

Chỉ so SAM với SGD trên cùng một loại GPU:

- **G4:** dùng 6 run của seed 1-3, sau khi đã xác nhận mọi epoch đều chạy G4 (Mục 5).
- **T4:** so `core_sgd_s0` với các epoch T4 của `core_sam_s0`.

Có tỷ lệ overhead trên hai loại GPU là một điểm có thể trình bày thêm. Không gộp hai tỷ lệ này thành một con số.

### 4.3 Peak GPU memory

`peak_gpu_memory_mb` lấy từ `torch.cuda.max_memory_allocated()` và là giá trị lớn nhất qua các phiên. Con số này chủ yếu phụ thuộc vào model và batch size. Tuy vậy, cuDNN có thể chọn thuật toán dùng vùng nhớ tạm khác nhau tùy GPU. Vì vậy cần ghi kèm loại GPU khi báo cáo.

## 5. Việc cần kiểm tra khi tổng hợp

- [ ] Xác nhận cột `gpu_name` trong `history.csv` của 8 core runs khớp với bảng ở Mục 1. Đặc biệt kiểm tra epoch đổi GPU của `core_sam_s0`, và kiểm tra seed 1-3 không có epoch nào bị cấp GPU khác khi resume.
- [ ] Xem `history.csv` của `core_sam_s0` quanh epoch đổi GPU. Train loss và validation accuracy không được nhảy vượt mức dao động bình thường giữa các epoch liền kề.
- [ ] Nếu seed 0 lệch hẳn so với seed 1-3, ghi phần cứng là một nguyên nhân có thể. Vẫn giữ seed 0 trong bảng tổng hợp; không loại run bất lợi.
- [ ] Bảng compute ghi rõ loại GPU cho từng con số, và tính time/epoch từ `history.csv`, không lấy từ `summary.json`.

## 6. Ảnh hưởng tới stage 2

`compute_sgd400_s0` được dùng để so với SAM 200 epochs ở mức tương đương về số lần tính gradient. Vì `core_sam_s0` chạy trên hai loại GPU, không so hai run này theo wall-clock time hoặc GPU-hour.

**Quyết định ngày 06/11/2026:**

1. Trục chính là số lần tính gradient. Accuracy của `compute_sgd400_s0` được so với mean ± SD của SAM và SGD 200 epochs trên cả 4 seed, vì accuracy của seed 0 vẫn hợp lệ (Mục 3).
2. Trục thời gian thực chỉ dùng dữ liệu G4: time/epoch của SGD và SAM lấy từ seed 1-3, và thời gian của `compute_sgd400_s0` nếu A chạy toàn bộ run này trên G4.

Chi tiết nằm trong [`2026-11-06-stage1-stage2-overview-vi.md`](2026-11-06-stage1-stage2-overview-vi.md), Mục 4.1.

## 7. Báo cáo time/epoch của stage 1

Time/epoch của stage 1 vẫn báo cáo được, với điều kiện mỗi con số chỉ lấy từ các epoch chạy trên cùng một loại GPU:

| GPU | SGD | SAM | Overhead SAM/SGD |
|---|---|---|---|
| G4 (bảng chính) | `core_sgd_s1`, `s2`, `s3`, mean ± SD | `core_sam_s1`, `s2`, `s3`, mean ± SD | Trên G4, 3 seed |
| T4 (bổ sung) | `core_sgd_s0`, toàn bộ 200 epochs | `core_sam_s0`, chỉ các epoch chạy T4 | Trên T4, 1 seed |

- `epoch_seconds` trong `history.csv` gồm thời gian train và đánh giá validation, không gồm thời gian ghi checkpoint. Validation tốn như nhau ở SGD và SAM, nên tỷ lệ overhead đo được thấp hơn mức 2 lần của riêng phần train. Báo cáo cần ghi rõ cách đo này.
- Epoch đầu của mỗi phiên Colab thường chậm hơn do khởi tạo CUDA và cuDNN. Dùng median hoặc bỏ epoch đầu của mỗi phiên, và áp dụng cùng một cách cho mọi run.
- Cùng là G4 nhưng tốc độ có thể dao động giữa các phiên Colab; SD giữa các seed thể hiện dao động này.

## 8. Nguồn đối chiếu

- `notebooks/sam_reproduction_vi.ipynb`: `seed_everything()`, `capture_random_state()`, `collect_environment()`, `run_experiment()`, `REQUIRED_SUMMARY_FIELDS`, `REQUIRED_HISTORY_COLUMNS`
- `docs/2026-10-23-sam-experiment-work-plan-vi.md`: Mục 6 (artifact), Giai đoạn 3 (tổng hợp core results), Mục 4.2 (run matrix)
- PyTorch, *CUDA semantics - TensorFloat-32 (TF32) on Ampere (and later) devices*: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
