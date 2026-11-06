# Tổng quan thực nghiệm: stage 1 và stage 2

- **Ngày ghi:** 06/11/2026
- **Trạng thái:** 8 core runs của stage 1 đã chạy xong; số liệu chưa được review chéo và tổng hợp. Stage 2: `rho` ablation của B đang chạy trên `run/rho-s1`; notebook `sam_stage2_extensions_vi_v1` và ba branch `run/sgd400-s0`, `run/noise40-s2`, `run/noise80-s2` đã sẵn sàng (chưa chạy trên GPU Colab).
- **Nguồn chính:** [`2026-10-23-sam-experiment-work-plan-vi.md`](2026-10-23-sam-experiment-work-plan-vi.md) (code_plan_v3). Note này tóm tắt và bổ sung, không thay thế work plan.

## 1. Bức tranh chung

Đồ án tái lập có kiểm soát một phần bài báo SAM (Foret et al., ICLR 2021) trên CIFAR-10 với WideResNet-28-10, 200 epochs. Mọi run dùng cùng notebook, cùng recipe và cùng validation split, nên kết quả của bốn thành viên ghép được thành một so sánh thống nhất.

| | Stage 1 - Core matrix | Stage 2 - Extensions |
|---|---|---|
| Câu hỏi | SAM có tốt hơn SGD không và tốn thêm bao nhiêu? | SAM tốt hơn vì sao, trong điều kiện nào, và có đáng chi phí không? |
| Số run | 8 (SGD và SAM × seed 0-3) | Tối đa 8 (3 `rho` + 1 SGD 400 epochs + 4 label noise) |
| Mức kết luận | Kết luận chính, mean ± SD trên 4 seed | Kết quả khám phá, mỗi cấu hình một seed |
| Vai trò | Baseline cho mọi so sánh | Phân tích sâu, cần cho mức điểm cao |

```mermaid
flowchart LR
    S1[Stage 1: 8 core runs] --> R[Review chéo artifact]
    R --> T[Tổng hợp core results - C]
    R --> N[Chốt quyết định, tạo notebook extensions]
    S1 -. B bắt đầu sớm từ 04/11 .-> E1[Rho ablation - B]
    N --> E2[SGD 400 epochs - A]
    N --> E3[Label noise 40% - C]
    N --> E4[Label noise 80% - D]
    T --> F[Khóa kết quả, slides, demo]
    E1 --> F
    E2 --> F
    E3 --> F
    E4 --> F
```

## 2. Stage 1 - Core matrix

### 2.1 Mục tiêu

Trả lời câu hỏi chính của đồ án:

> Trên CIFAR-10, với cùng model, data pipeline và training recipe, SAM có đạt test accuracy và generalization gap tốt hơn SGD hay không, và chi phí thời gian cùng bộ nhớ thay đổi như thế nào?

Bốn seed cho phép báo cáo mean ± SD để kết luận không phụ thuộc vào một run may mắn.

### 2.2 Việc cần làm khi có kết quả

1. **Review chéo artifact** (A↔C, B↔D) trước khi dùng bất kỳ con số nào: đối chiếu `config.json`, `history.csv` đủ 200 dòng, schema `summary.json`, `notebook_version = sam_reproduction_vi_v5` ở cả 8 run, và load được checkpoint. Run sai version hoặc thiếu artifact không được đưa vào tổng hợp.
2. **Bảng kết quả chính:**
   - Test accuracy mean ± sample SD cho SGD và SAM.
   - Chênh lệch `SAM - SGD` theo từng seed (paired, vì cùng seed thì cùng điều kiện).
   - Generalization gap. Notebook định nghĩa là accuracy trên train split với nhãn sạch, không augmentation, tại best checkpoint, trừ test accuracy.
   - Time/epoch, total time, peak GPU memory và tỷ lệ overhead của SAM, tách theo loại GPU (xem [`2026-11-06-core-s0-mixed-gpu-comparability-vi.md`](2026-11-06-core-s0-mixed-gpu-comparability-vi.md)).
3. **Hình:** learning curves của một cặp SGD/SAM cùng seed; paired plot SGD và SAM trên bốn seed kèm mean và error bar.
4. **Đối chiếu với paper:** đặt số của nhóm cạnh condition WRN-28-10, basic augmentation, 200 epochs trong Table 1 của paper. Phân biệt rõ số của paper và số nhóm tự chạy.
5. **Diễn giải trung thực:** bốn seed không đủ để dùng p-value. Báo cáo từng seed, mean và độ phân tán. Nếu SAM không hơn SGD, vẫn báo cáo đủ và giới hạn kết luận trong setting đã chạy.

### 2.3 Stage 1 là baseline cho stage 2

| Core run | Dùng trong stage 2 làm |
|---|---|
| `core_sgd_s1`, `core_sam_s1` | Đường tham chiếu SGD và điểm `rho = 0.05` của `rho` ablation |
| `core_sgd_s0`, `core_sam_s0` | Đối chứng cho SGD 400 epochs |
| `core_sgd_s2`, `core_sam_s2` | Điểm 0% noise của label-noise comparison |

Time/epoch đo được ở stage 1 cũng là căn cứ để kiểm tra lịch stage 2 còn khả thi trước ngày báo cáo 30/11.

## 3. Stage 2 - Extensions

### 3.1 Mục tiêu từng thí nghiệm

| # | Thí nghiệm | Câu hỏi | Run IDs | So với | Người chạy |
|---|---|---|---|---|---|
| 1 | `rho` ablation | SAM nhạy với `rho` đến mức nào; 0.05 có hợp lý trong setting của nhóm không? | `rho_001_s1`, `rho_010_s1`, `rho_020_s1` | `core_sgd_s1` (đường tham chiếu), `core_sam_s1` (`rho = 0.05`) | B, đang chạy |
| 2 | SGD 400 epochs | SAM hơn SGD có phải chỉ vì mỗi bước tốn gấp đôi gradient? SAM 200 epochs xấp xỉ SGD 400 epochs về số lần tính gradient. | `compute_sgd400_s0` | `core_sam_s0`, `core_sgd_s0` | A |
| 3 | Label noise 40% | SAM có bền hơn SGD khi nhãn train bị nhiễu? | `noise40_sgd_s2`, `noise40_sam_s2` | `core_sgd_s2`, `core_sam_s2` | C |
| 4 | Label noise 80% | Ở mức nhiễu nặng, khoảng cách SAM và SGD thay đổi thế nào? | `noise80_sgd_s2`, `noise80_sam_s2` | `core_*_s2`, `noise40_*_s2` | D |

Hình cần có cho slides:

- Test accuracy theo `rho`; SGD là đường ngang tham chiếu, không vẽ thành điểm "SAM `rho = 0`".
- Test accuracy của SGD và SAM ở 0%, 40% và 80% noise, cùng seed 2.
- So sánh SAM 200 epochs với SGD 200 và 400 epochs theo số lần tính gradient.

### 3.2 Quy tắc chung

- Mỗi extension chỉ có một seed nên chỉ được trình bày là **kết quả khám phá**, không dùng để khẳng định SAM tốt hơn một cách ổn định.
- Label-noise runs dùng chung seed 2 ở cả 0%, 40% và 80% để tách ảnh hưởng của noise rate khỏi khác biệt initialization.
- Thiếu tài nguyên thì bỏ CIFAR-100 trước, rồi bỏ label noise 80%. Không hy sinh tính nhất quán của pipeline để chạy thêm extension.

### 3.3 Hiện trạng notebook

| Notebook | Version | Hỗ trợ | Ghi chú |
|---|---|---|---|
| `notebooks/sam_reproduction_vi.ipynb` | `sam_reproduction_vi_v5` | `RUN_PLAN` thuộc `smoke`, `core` | Run ID có dạng `{RUN_PLAN}_{method}_s{SEED}`; `LABEL_NOISE`, `EPOCHS`, `METHODS` là hằng số recipe |
| `notebooks/sam_stage2_reproduction_vi.ipynb` | `sam_stage2_reproduction_vi_v1` | `RUN_PLAN` thuộc `smoke`, `rho`; chỉ chạy SAM | B đang resume trên `run/rho-s1` |

Label noise đã có sẵn trong pipeline (`NoisySubset`): symmetric noise, mỗi nhãn bị chọn được đổi sang một lớp khác ngẫu nhiên, xác định theo `SEED`. Noise chỉ áp lên train split; validation, test và phép đo train accuracy cho generalization gap dùng nhãn sạch. Cột `train_accuracy` trong `history.csv` được đo trong lúc train nên tính trên nhãn nhiễu.

Cosine schedule dùng `T_max = EPOCHS`, nên SGD 400 epochs có learning-rate schedule trải đều trên 400 epochs.

### 3.4 Notebook cho các extension còn lại

Không thể thêm plan mới vào notebook `rho` mà không tăng `NOTEBOOK_VERSION`, và tăng version sẽ làm resume của B trên `run/rho-s1` từ chối tiếp tục. Không thể dùng notebook core vì `RUN_PLAN` chỉ nhận `smoke` và `core`, và branch chỉ được khác `main` ở `RUN_PLAN` và `SEED`.

Vì vậy ba extension còn lại dùng notebook mới `notebooks/sam_stage2_extensions_vi.ipynb` (`sam_stage2_extensions_vi_v1`), nhân bản từ `sam_reproduction_vi_v5`. Notebook chỉ khác bản core ở các cell cấu hình và mô tả; mọi cell training, data, checkpoint và artifact giống hệt từng byte. Bảng `RUN_PLANS` trong cell cấu hình định nghĩa:

| `RUN_PLAN` | Methods | Label noise | Epochs | Run IDs | Branch |
|---|---|---:|---:|---|---|
| `smoke` | SGD, SAM (WRN-16-1, 1.024 ảnh) | 40% | 5 | `smoke_ext_sgd_s<SEED>`, `smoke_ext_sam_s<SEED>` | `main` |
| `sgd400` | SGD | 0% | 400 | `compute_sgd400_s0` | `run/sgd400-s0` |
| `noise40` | SGD, SAM | 40% | 200 | `noise40_sgd_s2`, `noise40_sam_s2` | `run/noise40-s2` |
| `noise80` | SGD, SAM | 80% | 200 | `noise80_sgd_s2`, `noise80_sam_s2` | `run/noise80-s2` |

Plan `smoke` có 40% label noise để smoke test đi qua nhánh label noise, vốn chưa được core run nào dùng.

Kiểm tra ngày 06/11 trên CPU với Drive giả lập (`tools/run_notebook_locally.py`, PyTorch 2.15.0). Chưa chạy trên GPU Colab.

| Kiểm tra | Kết quả |
|---|---|
| Plan `smoke` seed 0 chạy end-to-end | `NOTEBOOK_OK`; cả hai run có `artifact_status = OK` và `checkpoint_inference = OK` |
| Tương đương với notebook core: bản sao có plan `smoke` đặt label noise 0% và run ID của core | `history.csv` và các chỉ số trong `summary.json` trùng với smoke test của notebook core; chỉ khác `notebook_version` và thời gian |
| Ngắt giữa run SAM (sau epoch 2/5) rồi Run all lại | Bỏ qua run SGD đã xong, resume SAM từ epoch 3; `history.csv` và chỉ số trùng với run không bị ngắt |
| Cell cấu hình của từng plan | Run IDs, method, label noise và epochs đúng như bảng trên; `RUN_PLAN = "core"` bị từ chối; plan khác `smoke` bị từ chối khi không có GPU |
| `NoisySubset` trên 45.000 nhãn, seed 2 | Đúng 40,00% và 80,00% nhãn bị đổi; SGD và SAM cùng seed nhận cùng tập nhãn nhiễu |

## 4. Quyết định đã chốt và điểm cần lưu ý

### 4.1 Quyết định ngày 06/11/2026

1. **`rho = 0.05` cho label-noise runs ở cả 40% và 80%**, giống core run. Mỗi mức noise khi đó chỉ khác nhau ở tỷ lệ noise. Paper dùng `rho = 0.1` cho phần lớn mức noise và 0.05 ở mức 80% (xem [`2026-10-12-sam-paper-summary-vi.md`](2026-10-12-sam-paper-summary-vi.md), Mục 9.4), và giá trị đó được tune trong setup của paper. Báo cáo ghi kết quả là "SAM với `rho` mặc định, không tune theo noise" và nêu rằng khoảng cách SAM-SGD đo được có thể nhỏ hơn paper.
2. **So `compute_sgd400_s0` với SAM 200 epochs chủ yếu theo số lần tính gradient** (SGD 200 epochs = 200 đơn vị, SAM 200 epochs = 400, SGD 400 epochs = 400).
   - Accuracy: so `compute_sgd400_s0` với mean ± SD của SAM và SGD 200 epochs trên cả 4 seed. Accuracy của `core_sam_s0` vẫn hợp lệ dù chạy trên hai loại GPU. Cặp `core_sgd_s0` và `compute_sgd400_s0` cùng seed cho thấy tác động của việc tăng gấp đôi số epoch.
   - Thời gian thực: chỉ dùng dữ liệu G4, gồm time/epoch của seed 1-3 và `compute_sgd400_s0` nếu run này chạy toàn bộ trên G4.
   - `compute_sgd400_s0` chỉ có một run, nên chỉ kết luận nó nằm trong hay ngoài khoảng của SAM 200 epochs trên 4 seed.

### 4.2 Điểm cần lưu ý khi diễn giải

1. **Clean validation trong label-noise runs.** Notebook chọn checkpoint bằng validation nhãn sạch, nên SGD được "dừng sớm" trước khi học thuộc nhãn sai. Đây là giả định thuận lợi so với thực tế và có thể làm khoảng cách SAM-SGD nhỏ đi. Ngoài số tại best checkpoint, nên báo cáo thêm test accuracy tại `last_checkpoint.pt` khi tổng hợp. Cách này không chọn checkpoint bằng test set, chỉ mô tả thêm. Cần kiểm tra paper báo cáo theo checkpoint nào trước khi đặt số cạnh paper.
2. **Định nghĩa label noise.** `NoisySubset` luôn đổi nhãn bị chọn sang một lớp **khác** lớp đúng. Ở 80% noise, lớp đúng còn 20% và mỗi lớp sai khoảng 8,9%. Một số tài liệu định nghĩa symmetric noise là gán lại nhãn ngẫu nhiên đều trên 10 lớp, có thể trùng lớp đúng, nên mức nhiễu thực tế thấp hơn tỷ lệ danh nghĩa. Cần kiểm tra paper dùng định nghĩa nào trước khi so số trực tiếp; nếu khác thì nêu trong phần hạn chế.
3. **`train_accuracy` trong `history.csv` của label-noise runs tính trên nhãn nhiễu**, còn `checkpoint_train_accuracy` và `generalization_gap` tính trên nhãn sạch.

### 4.3 Việc còn mở

1. **Quota và dung lượng.** A, C, D xác nhận còn đủ quota GPU và dung lượng Drive (`sgd400` khoảng 1,1 GB; mỗi cặp label noise khoảng 2,3 GB). Nên chạy toàn bộ một run trên cùng loại GPU nếu có thể và luôn kiểm tra cột `gpu_name`.
2. **Lịch.** Work plan đặt extensions vào 11-17/11 và slides từ 18/11. Ước lượng lại từ time/epoch thực tế của stage 1; `compute_sgd400_s0` tốn khoảng thời gian của một core SAM run.

## 5. Thứ tự thực hiện

1. Review chéo 8 core runs và kiểm tra cột `gpu_name`.
2. ~~Chốt các quyết định ở Mục 4.~~ Đã chốt ngày 06/11.
3. ~~Tạo `sam_stage2_extensions_vi.ipynb` trên `main`, smoke test local, rồi tạo ba branch run.~~ Đã xong ngày 06/11 (chưa chạy trên GPU Colab).
4. A, C, D chạy extension; B tiếp tục `rho` ablation.
5. Song song: C tổng hợp core results theo Mục 2.2.
6. Tổng hợp extension, khóa kết quả, làm slides và demo.
